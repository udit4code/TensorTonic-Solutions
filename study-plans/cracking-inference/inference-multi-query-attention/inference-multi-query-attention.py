import math
import torch


class SafeSoftmax:
    """
    A utility class to compute numerically stable softmax.
    
    Why do we need 'safe' softmax?
    Standard exponential function torch.exp(x) blows up to infinity very easily 
    if x is a large positive number (e.g., e^100 is massive). By shifting the input 
    so the maximum value is 0 (x - max), all values become <= 0, ensuring that 
    torch.exp() output is always between 0 and 1. This prevents NaN errors.
    """
    def __init__(self, dim: int = -1):
        self.dim = dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Find the maximum value along the target dimension.
        # keepdim=True ensures the dimension isn't squashed, allowing clean broadcasting later.
        # Note: torch.max returns a namedtuple (values, indices), so we extract .values
        max_x = torch.max(x, dim=self.dim, keepdim=True).values
        
        # Step 2: Shift values so the largest value becomes 0 (x - max <= 0)
        shifted_x = x - max_x
        
        # Step 3: Exponentiate the shifted values (e^(x - max) <= e^0 = 1)
        exp_x = torch.exp(shifted_x)
        
        # Step 4: Sum the exponentials along the dimension to get the normalization denominator
        sum_exp_x = torch.sum(exp_x, dim=self.dim, keepdim=True)
        
        # Step 5: Divide to get valid probabilities that sum up to 1 along the dimension
        return exp_x / sum_exp_x


class MultiQueryAttention:
    """
    Implements Multi-Query Attention (MQA).
    
    Key Difference from Standard Multi-Head Attention (MHA):
    - In MHA, each attention head has its own independent Query ($Q$), Key ($K$), and Value ($V$) projections.
    - In MQA, every query head still has its own independent Query projection, but ALL heads 
      share the exact same single Key ($K$) and Value ($V$) projection.
    
    The Tradeoff:
    - MQA has slightly less representational power than MHA.
    - However, the size of the KV-cache shrinks drastically during generation, saving massive 
      amounts of GPU memory and speeding up token generation!
    """
    def __init__(
        self,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        w_o: torch.Tensor,
        num_query_heads: int,
        causal: bool = False,
    ):
        """
        Initializes the MQA layer by storing the weight matrices and configuration settings.
        """
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v
        self.w_o = w_o
        self.num_query_heads = num_query_heads
        self.causal = causal
        
        # Instantiate our safe softmax helper
        self.safe_softmax = SafeSoftmax(dim=-1)

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of Multi-Query Attention.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Step 0: Extract tensor dimensions
        # B = batch size, T = sequence length (number of tokens), D = model hidden dimension
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Ensure our hidden dimension can be evenly split across our query heads
        assert d_model % self.num_query_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_query_heads ({self.num_query_heads})"
        )
        
        # head_dim ($d_h$) is the width of each individual attention head
        head_dim = d_model // self.num_query_heads

        # Step 1: Query Projection
        # Matrix multiplication: (Batch, Seq, D_model) @ (D_model, D_model) -> (Batch, Seq, D_model)
        q = hidden_states @ self.w_q

        # Step 2: Split Query into Multiple Heads
        # Shape transition: (B, T, D) -> reshape to (B, T, H, d_h) -> transpose to (B, H, T, d_h)
        # H = number of query heads, d_h = head dimension
        q = q.reshape(batch_size, seq_len, self.num_query_heads, head_dim).transpose(1, 2)

        # Step 3: Shared Key and Value Projections
        # Notice that unlike MQA queries, we project into a single head dimension ($d_h$) instead of full $D$.
        # (B, T, D) @ (D, d_h) -> (B, T, d_h)
        k = hidden_states @ self.w_k
        v = hidden_states @ self.w_v

        # Step 4: Add a dummy head dimension (1 head) to K and V to allow broadcasting with Q later
        # Shape transition: (B, T, d_h) -> reshape to (B, T, 1, d_h) -> transpose to (B, 1, T, d_h)
        k = k.reshape(batch_size, seq_len, 1, head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, 1, head_dim).transpose(1, 2)

        # Current Tensor Shapes Status:
        # q: (Batch, Query_Heads, Seq_Len, Head_Dim) -> (B, H, T, d_h)
        # k: (Batch, 1,           Seq_Len, Head_Dim) -> (B, 1, T, d_h)
        # v: (Batch, 1,           Seq_Len, Head_Dim) -> (B, 1, T, d_h)

        # Step 5: Compute Attention Scores and Scale by Square Root of Head Dimension
        # Matrix multiplication: (B, H, T, d_h) @ (B, 1, d_h, T) -> (B, H, T, T)
        # PyTorch automatically broadcasts the size-1 head dimension of 'k' across all 'H' query heads!
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(head_dim)

        # Step 6: Apply Causal Mask (if enabled)
        # Causal masking prevents future tokens from looking backward into past tokens during autoregressive training.
        if self.causal:
            # Create a 2D grid of row indices and column indices.
            # .view(seq_len, 1) makes row_index look like a column vector: [[0], [1], [2], ...]
            # .view(1, seq_len) makes col_index look like a row vector:    [[0, 1, 2, ...]]
            row_index = torch.arange(seq_len, device=hidden_states.device).view(seq_len, 1)
            col_index = torch.arange(seq_len, device=hidden_states.device).view(1, seq_len)
            
            # A boolean matrix where True means "this position is in the future"
            mask = row_index < col_index
            
            # Wherever the mask is True, replace the attention score with negative infinity (-inf).
            # When softmax is later applied to -inf, it turns into 0 probability!
            neg_inf = torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype)
            scores = torch.where(mask, neg_inf, scores)

        # Step 7: Convert raw scores into [0, 1] attention probabilities using SafeSoftmax
        # Shape remains (B, H, T, T)
        attention_weights = self.safe_softmax(scores)

        # Step 8: Compute Weighted Sum over Values (V)
        # Since MQA shares a single V head, PyTorch broadcasts the (B, 1, T, d_h) shape 
        # across all query heads.
        # (B, H, T, T) @ (B, 1, T, d_h) -> (B, H, T, d_h)
        head_outputs = attention_weights @ v

        # Step 9: Concatenate All Query Heads Back Together
        # Shape transition: (B, H, T, d_h) -> transpose to (B, T, H, d_h) -> reshape to (B, T, D)
        concatenated_head_outputs = (
            head_outputs.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, seq_len, d_model)
        )

        # Step 10: Final Output Projection
        # Project combined head outputs back through the output weight matrix $W_o$
        # (B, T, D) @ (D, D) -> (B, T, D)
        output = concatenated_head_outputs @ self.w_o

        return output
        
def multi_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns an attention tensor with the same shape as hidden_states.
    """
    # Instantiate the object-oriented MultiQueryAttention layer with provided weights and configs
    layer = MultiQueryAttention(
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        num_query_heads=num_query_heads,
        causal=causal
    )
    
    # Invoke the layer directly using __call__
    return layer(hidden_states)

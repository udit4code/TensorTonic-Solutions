-- Write your SQL query here
Select 
    p.product_name As name,
    p.category As category,
    p.unit_price * p.units_in_stock as inventory_value
From products As P
def validate_records(records: list, schema: list) -> list:
    """
    Validates warehouse records against an ordered schema.
    Returns a list of result dictionaries containing record_index, is_valid, and errors.
    """
    results = []
    
    for record_index, record in enumerate(records):
        errors = []
        
        for entry in schema:
            col = entry["column"]
            expected_type = entry["type"]
            nullable = entry["nullable"]
            
            # Check 1 : Check missing values
            if col not in record:
                errors.append(f"{col}: missing")
                continue
            
            # Check 2 : Check nullability
            val = record[col]
            if val is None:
                if nullable:
                    continue  # Valid nullable null, skip later checks for this column
                else:
                    errors.append(f"{col}: null")
                    continue
            
            # Check 3 : Check type
            actual_type = type(val).__name__
            type_valid = False
            
            if expected_type == "int":
                if isinstance(val, int) and type(val) is not bool:
                    type_valid = True
            elif expected_type == "float":
                # Accepts Python integers and floats, but not booleans
                if isinstance(val, (int, float)) and type(val) is not bool:
                    type_valid = True
            elif expected_type == "str":
                if isinstance(val, str):
                    type_valid = True
            
            if not type_valid:
                errors.append(f"{col}: expected {expected_type}, got {actual_type}")
                continue  # Skip range check if type check failed
                
            # Check 4 : Check range (min / max bounds)
            if "min" in entry and entry["min"] is not None:
                if val < entry["min"]:
                    errors.append(f"{col}: out of range")
                    continue
            if "max" in entry and entry["max"] is not None:
                if val > entry["max"]:
                    errors.append(f"{col}: out of range")
                    continue
                    
        results.append({
            "record_index": record_index,
            "is_valid": len(errors) == 0,
            "errors": errors
        })
        
    return results
SELECT name, subject, score
FROM students
ORDER BY 
    -- score IS NULL ASC evaluates to 0 for valid scores and 1 for NULL scores, ensuring all non-null scores come first and NULL scores are pushed to the bottom.
    score IS NULL ASC, 
    score DESC, 
    -- name IS NULL ASC ensures valid names come before NULL names.
    name IS NULL ASC, 
    name ASC;
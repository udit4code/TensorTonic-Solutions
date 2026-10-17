SELECT name, subject, score
FROM students
ORDER BY 
    score IS NULL ASC, 
    score DESC, 
    name IS NULL ASC, 
    name ASC;
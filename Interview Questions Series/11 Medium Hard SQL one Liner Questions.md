Absolutely — here are **100 more medium–hard SQL one-liner questions**, each with a tiny, simple hint and a tiny sprinkle of fun.

---

### **50 Medium–Hard SQL Questions (With Hints)**

1. **How do you remove duplicate rows while keeping the latest record?** 
    #### → (Hint: Window magic)
2. **How can you pivot rows into columns dynamically?** 
    #### → (Hint: Columns that appear on the fly)
3. **How do you unpivot multiple columns into rows?** 
    #### → (Hint: Spread becomes skinny)
4. **How do you find the second-highest salary without using MAX twice?** 
    #### → (Hint: OFFSET helps)
5. **How do you get the nth highest value in a column?** 
    #### → (Hint: Ranking helps)
6. **How do you identify gaps in a sequence of numbers?** 
    #### → (Hint: Compare with row numbers)
7. **How do you detect overlapping date ranges?** 
    #### → (Hint: Start < End trouble)
8. **How do you split a comma-separated string into rows?** 
    #### → (Hint: Explode the list)
9. **How do you join only the latest record from a history table?** 
    #### → (Hint: Row_number = 1)
10. **How do you compare two tables and find mismatched rows?** 
    #### → (Hint: Full outer join detective)
---
11. **How do you update a table using data from another table?** 
    #### → (Hint: Update + join)
12. **How do you find customers who bought the same product twice in a row?** 
    #### → (Hint: Lag behind)
13. **How do you calculate running totals?** 
    #### → (Hint: Cumulative windows)
14. **How do you get the most recent non-null value?** 
    #### → (Hint: Lag with filtering)
15. **How do you find islands and gaps in time-based data?** 
    #### → (Hint: Lead and lag)
16. **How do you detect duplicate groups, not duplicate rows?** 
    #### → (Hint: Count > 1 trick)
17. **How do you create a recursive hierarchy query?** 
    #### → (Hint: CTE climbs trees)
18. **How do you find rows where a column changed compared to previous rows?** 
    #### → (Hint: Lag again!)
19. **How do you remove records appearing more than N times?** 
    #### → (Hint: Group, then chop)
20. **How do you get rows with the maximum value per group?** 
    #### → (Hint: Partition by group)

---

21. **How do you join on multiple conditions including range matches?** 
    #### → (Hint: Between joins)
22. **How do you check if two tables have exactly the same data?** 
    #### → (Hint: Except or minus)
23. **How do you detect when a sequence resets?** 
    #### → (Hint: Row_number with partitions)
24. **How do you find the longest streak of consecutive days?** 
    #### → (Hint: Date difference trick)
25. **How do you generate a date calendar on the fly?** 
    #### → (Hint: Recursive CTE again)
26. **How do you remove leading zeros from a string?** 
    #### → (Hint: Cast then convert)
27. **How do you return only rows that appear in all groups?** 
    #### → (Hint: Having count distinct)
28. **How do you filter groups based on conditions inside them?** 
    #### → (Hint: Having is boss)
29. **How do you rank ties without skipping numbers?** 
    #### → (Hint: Dense_rank)
30. **How do you rank ties while skipping numbers?** 
    #### → (Hint: Rank)
---
31. **How do you list employees who earn more than their department averages?** 
    #### → (Hint: Subquery challenge)
32. **How do you create rolling monthly averages?** 
    #### → (Hint: Frame clauses)
33. **How do you find customers who never bought anything?** 
    #### → (Hint: Anti join)
34. **How do you find customers who bought everything in a category?** 
    #### → (Hint: Division logic)
35. **How do you identify the “top 1 per category” when ties exist?** 
    #### → (Hint: Rank with filters)
36. **How do you find the 3 most sold products per region?** 
    #### → (Hint: Limit inside window)
37. **How do you detect cycles in hierarchical data?** 
    #### → (Hint: Recursive CTE with stop)
38. **How do you sample random records efficiently?** 
    #### → (Hint: Order by random is expensive)
39. **How do you count distinct values but ignore nulls?** 
    #### → (Hint: Distinct does it)
40. **How do you replace nulls with values from another column?** 
    #### → (Hint: Coalesce)

---

41. **How do you combine multiple rows into a single comma-separated list?** 
    #### → (Hint: String aggregate)
42. **How do you validate if a string is numeric?** 
    #### → (Hint: Use regex or try_cast)
43. **How do you find overlapping subscriptions per user?** 
    #### → (Hint: Date overlaps again)
44. **How do you fetch one row per group randomly?** 
    #### → (Hint: Random + window)
45. **How do you delete duplicates but keep the lowest ID?** 
    #### → (Hint: Row_number = 1 survives)
46. **How do you detect missing foreign key references?** 
    #### → (Hint: Left join + null check)
47. **How do you compute year-over-year percentage change?** 
    #### → (Hint: Lag yearly)
48. **How do you flatten nested JSON fields?** 
    #### → (Hint: Cross apply openjson)
49. **How do you join on a computed value like the first letter of a name?** 
    #### → (Hint: Join expressions)
50. **How do you find the earliest timestamp per user but only on weekdays?** 
    #### → (Hint: Filter + grouping)

---

# **100 More Medium–Hard SQL Questions (With Hints)**

### **1–20: Joins, Grouping & Aggregations**

1. **How do you find rows that match on one column but differ on another?** 
    #### → (Hint: Same same, but different)
2. **How do you list groups where the average exceeds the median?** 
    #### → (Hint: Median needs a window trick)
3. **How do you identify groups where no value repeats?** 
    #### → (Hint: Count distinct vs count)
4. **How do you compute weighted averages?** 
    #### → (Hint: Multiply before dividing)
5. **How do you find rows belonging to the top 10% of values?** 
    #### → (Hint: Percent_rank)
6. **How do you find groups with continuous sequences of IDs?** 
    #### → (Hint: ID difference trick)
7. **How do you show customers who bought more than one type of product?** 
    #### → (Hint: Count distinct)
8. **How do you list groups whose total matches the grand total?** 
    #### → (Hint: Having = total)
9. **How do you get the min and max values from the same group in one scan?** 
    #### → (Hint: Aggregate both)
10. **How do you find each group's most common value?** 
    #### → (Hint: Mode with count)
11. **How do you detect groups with outliers?** 
    #### → (Hint: Stddev or IQR)
12. **How do you count only rows that meet a condition?** 
    #### → (Hint: SUM with CASE)
13. **How do you calculate conditional percentages?** 
    #### → (Hint: Divide conditional counts)
14. **How do you list the top-k items per group but without ties?** 
    #### → (Hint: Row_number)
15. **How do you make groups of size N from a table?** 
    #### → (Hint: Use row_number chunks)
16. **How do you compute moving min/max values?** 
    #### → (Hint: Window frame)
17. **How do you calculate the geometric mean?** 
    #### → (Hint: Multiply then nth root)
18. **How do you find rows belonging to incomplete groups?** 
    #### → (Hint: Having count < expected)
19. **How do you get the average of the top 3 values per user?** 
    #### → (Hint: Rank then avg)
20. **How do you list the groups with no matching rows in another table?** 
    #### → (Hint: Anti join)

---

### **21–40: Window Functions**

21. **How do you get the previous non-null value in a sequence?** 
    #### → (Hint: Ignore nulls)
22. **How do you compute the time between events per user?** 
    #### → (Hint: Lag timestamps)
23. **How do you rank events by custom ordering?** 
    #### → (Hint: Order by expressions)
24. **How do you get cumulative unique counts?** 
    #### → (Hint: Distinct windows… tricky)
25. **How do you compare current row to the average of last 5 rows?** 
    #### → (Hint: Frame rows)
26. **How do you get the earliest event of each quarter?** 
    #### → (Hint: Partition by quarter)
27. **How do you find sudden spikes in values?** 
    #### → (Hint: Compare with lag)
28. **How do you compute half-life decay values row by row?** 
    #### → (Hint: Recursive + window)
29. **How do you track “running max minus running min”?** 
    #### → (Hint: Two windows)
30. **How do you label streaks of identical values?** 
    #### → (Hint: Row_number difference)
31. **How do you detect when a rolling sum crosses a threshold?** 
    #### → (Hint: Window sum + compare)
32. **How do you find values that drop compared to last month?** 
    #### → (Hint: Lag by interval)
33. **How do you create a moving z-score?** 
    #### → (Hint: Mean & stddev window)
34. **How do you get “time since first event”?** 
    #### → (Hint: Min timestamp window)
35. **How do you compute cumulative median?** 
    #### → (Hint: Hard — nested windows)
36. **How do you assign row numbers only within certain conditions?** 
    #### → (Hint: Partition on CASE)
37. **How do you calculate rolling distinct counts?** 
    #### → (Hint: Precompute mapping)
38. **How do you detect every time a value resets to zero?** 
    #### → (Hint: Lag compare)
39. **How do you calculate “current rank vs previous rank”?** 
    #### → (Hint: Lag the rank)
40. **How do you compute running unique sums like “new categories seen”?** 
    #### → (Hint: Boolean accumulators)

---

### **41–60: Subqueries, Advanced Filtering**

41. **How do you find customers whose total spending is above the overall median?** 
    #### → (Hint: Subquery + compare)
42. **How do you find rows that don’t belong to any subquery results?** 
    #### → (Hint: Not exists)
43. **How do you get rows where a value is the minimum but only when the group meets a condition?** 
    #### → (Hint: Conditional having)
44. **How do you fetch only the earliest row that matches multiple filters?** 
    #### → (Hint: Order then limit)
45. **How do you find records in table A that match multiple rows in table B?** 
    #### → (Hint: Group by key)
46. **How do you get the largest gap between two events?** 
    #### → (Hint: Lead difference)
47. **How do you list values that occur exactly once globally?** 
    #### → (Hint: Having count = 1)
48. **How do you detect “orphaned” child rows?** 
    #### → (Hint: No parent match)
49. **How do you find pairs of users who share at least 3 common items?** 
    #### → (Hint: Self-join + group)
50. **How do you detect when a user changes their category?** 
    #### → (Hint: Lag category)
51. **How do you find rows that match the top value per group but non-uniquely?** 
    #### → (Hint: Max join)
52. **How do you return all rows except the last N?** 
    #### → (Hint: Offset or anti-window)
53. **How do you select rows that appear in *exactly* two categories?** 
    #### → (Hint: Count distinct)
54. **How do you find users who have gaps in activity > X days?** 
    #### → (Hint: Date diff with lag)
55. **How do you find rows matching the most frequent value?** 
    #### → (Hint: Mode)
56. **How do you get customers who bought A before B?** 
    #### → (Hint: Min dates compare)
57. **How do you list rows where the next row is earlier in time?** 
    #### → (Hint: Out-of-order detection)
58. **How do you find the first purchase after a specific event?** 
    #### → (Hint: Min with filter)
59. **How do you return rows only if the subquery count equals N?** 
    #### → (Hint: Having)
60. **How do you find rows whose value is greater than the average of their neighbors?** 
    #### → (Hint: Lag + lead)

---

### **61–80: Strings, Dates, JSON, and Functions**

61. **How do you extract the numeric part from alphanumeric strings?** 
    #### → (Hint: Regex rescue)
62. **How do you find palindromic strings in a table?** 
    #### → (Hint: Reverse = original)
63. **How do you remove repeated characters in a string?** 
    #### → (Hint: Regex replace)
64. **How do you generate a running concatenated string?** 
    #### → (Hint: String_agg window)
65. **How do you check if a string contains only vowels?** 
    #### → (Hint: Regex check)
66. **How do you correct invalid dates stored as text?** 
    #### → (Hint: Try-convert)
67. **How do you find rows where timestamps overlap but not fully?** 
    #### → (Hint: Partial overlap logic)
68. **How do you calculate week numbers respecting ISO rules?** 
    #### → (Hint: ISO week function)
69. **How do you merge two JSON arrays in SQL?** 
    #### → (Hint: JSON array combine)
70. **How do you extract nested key-value pairs from JSON columns?** 
    #### → (Hint: Cross apply)
71. **How do you detect missing keys in semi-structured data?** 
    #### → (Hint: JSON existence check)
72. **How do you convert epoch milliseconds to timestamps?** 
    #### → (Hint: Add seconds)
73. **How do you get the last day of each month?** 
    #### → (Hint: Date functions)
74. **How do you group by hour when timestamp includes milliseconds?** 
    #### → (Hint: Truncate)
75. **How do you filter rows where the date falls on weekends?** 
    #### → (Hint: Day of week)
76. **How do you calculate the number of business days between two dates?** 
    #### → (Hint: Calendar table)
77. **How do you check if two strings are anagrams?** 
    #### → (Hint: Sort characters)
78. **How do you extract URLs from messy text?** 
    #### → (Hint: Regex sniper)
79. **How do you group by transformed string values like lowercase names?** 
    #### → (Hint: Group by expression)
80. **How do you detect invalid JSON in a column?** 
    #### → (Hint: Try_parse)

---

### **81–100: Performance, Optimization & Architecture**

81. **How do you detect missing indexes that would speed up a query?** 
    #### → (Hint: Explain plan clue)
82. **How do you rewrite correlated subqueries to improve speed?** 
    #### → (Hint: Join instead)
83. **How do you reduce sorting in window functions?** 
    #### → (Hint: Partition wisely)
84. **How do you avoid scanning an entire table when filtering by a non-indexed column?** 
    #### → (Hint: Create or use index)
85. **How do you rewrite queries to avoid the dreaded “SELECT *”?** 
    #### → (Hint: Pick columns)
86. **How do you detect when an index is never used?** 
    #### → (Hint: Index usage stats)
87. **How do you speed up “top-k” per group queries?** 
    #### → (Hint: Partial sort)
88. **How do you reduce join explosion in many-to-many joins?** 
    #### → (Hint: Pre-aggregate)
89. **How do you rewrite OR conditions for better performance?** 
    #### → (Hint: Union all)
90. **How do you detect cardinality misestimation?** 
    #### → (Hint: Explain plan off)
91. **How do you optimise wide table scans with many unused columns?** 
    #### → (Hint: Column pruning)
92. **How do you reduce the cost of large IN lists?** 
    #### → (Hint: Temp table or join)
93. **How do you handle NULLs in join keys without performance drop?** 
    #### → (Hint: Coalesce carefully)
94. **How do you check if your join order is efficient?** 
    #### → (Hint: Check plan)
95. **How do you avoid window function recomputation?** 
    #### → (Hint: Subquery reuse)
96. **How do you maintain summary tables efficiently?** 
    #### → (Hint: Incremental loads)
97. **How do you speed up “distinct” queries on big datasets?** 
    #### → (Hint: Hash aggregations)
98. **How do you ensure partition pruning happens?** 
    #### → (Hint: Filter on partition column)
99. **How do you detect if your query is spilling to disk?** 
    #### → (Hint: Explain/monitor)
100. **How do you decide between indexing and partitioning?** 
    #### → (Hint: Filter vs volume)

---

# **100 More Medium–Hard SQL Questions (Fresh & Non-Repeated)**

### **1–20: Joins, Grouping, Set Logic**

1. **How do you find rows that appear in table A but not at all in table B?** 
    #### → (Hint: Anti-join)
2. **How do you show only rows that match in both tables with different column names?** 
    #### → (Hint: Explicit join mapping)
3. **How do you join two tables when the join key can be in one of two columns?** 
    #### → (Hint: OR join… carefully)
4. **How do you match rows where one table stores ranges and the other stores single values?** 
    #### → (Hint: Between join)
5. **How do you detect one-to-many relationships accidentally turning into many-to-many?** 
    #### → (Hint: Unexpected row explosion)
6. **How do you join on the nearest date?** 
    #### → (Hint: Min(abs(date diff)))
7. **How do you find groups whose minimum appears in multiple rows?** 
    #### → (Hint: Join with subquery)
8. **How do you compute group-level statistics excluding the current row?** 
    #### → (Hint: Window frame skip)
9. **How do you union two sets but keep duplicates?** 
    #### → (Hint: UNION ALL)
10. **How do you compare two tables row-by-row including null differences?** 
    #### → (Hint: Full join + coalesce)
11. **How do you find rows that exist in both tables but with different values?** 
    #### → (Hint: Join + inequality)
12. **How do you list groups that have at least one NULL and at least one NOT NULL?** 
    #### → (Hint: Boolean grouping)
13. **How do you check if two queries produce identical result sets?** 
    #### → (Hint: EXCEPT both ways)
14. **How do you detect when a group violates a uniqueness rule?** 
    #### → (Hint: Count > 1)
15. **How do you implement relational division?** 
    #### → (Hint: “All of” logic)
16. **How do you find groups missing exactly one expected item?** 
    #### → (Hint: Expected count minus actual)
17. **How do you join tables where one key is case-sensitive and the other is not?** 
    #### → (Hint: Apply function)
18. **How do you detect multiple parent matches for a child row?** 
    #### → (Hint: Count per child)
19. **How do you join when one table stores composite keys in a single column?** 
    #### → (Hint: Split + join)
20. **How do you return matching rows even if the join column contains trailing spaces?** 
    #### → (Hint: Trim)

---

### **21–40: Window Functions, Ranking, Time Series**

21. **How do you find the largest interval between successive timestamps?** 
    #### → (Hint: Lead gap)
22. **How do you return only rows where the value jumps by >50% compared to previous row?** 
    #### → (Hint: Lag + ratio)
23. **How do you assign rank but restart ranking whenever value resets to zero?** 
    #### → (Hint: Partition by running group)
24. **How do you create a window that looks only forward, not backward?** 
    #### → (Hint: Lead frame)
25. **How do you compare each row with the average of the next 3 rows?** 
    #### → (Hint: Following window)
26. **How do you calculate rolling percentiles?** 
    #### → (Hint: Window with percentile_cont)
27. **How do you compute “days since previous activity” per user?** 
    #### → (Hint: Lag timestamps)
28. **How do you detect when a streak of increasing values breaks?** 
    #### → (Hint: Value < lag)
29. **How do you compute a moving weighted sum?** 
    #### → (Hint: Frame + math)
30. **How do you generate an index that restarts every time a category changes?** 
    #### → (Hint: Row number + change detection)
31. **How do you calculate the slope of a time series using SQL only?** 
    #### → (Hint: Regression functions)
32. **How do you detect monotonic trends per group?** 
    #### → (Hint: LEAST + GREATEST)
33. **How do you compute rolling correlation between two columns?** 
    #### → (Hint: Window + stats)
34. **How do you detect the first row where a rolling sum exceeds a threshold?** 
    #### → (Hint: Running total + filter)
35. **How do you assign session IDs based on gaps in timestamps?** 
    #### → (Hint: Compare with lag)
36. **How do you compute trailing 12-month sums ignoring partial months?** 
    #### → (Hint: Range between 12 months)
37. **How do you remove rows that are duplicates in everything except timestamp?** 
    #### → (Hint: Row_number)
38. **How do you detect rows where the timestamp jumps backward?** 
    #### → (Hint: Lag comparison)
39. **How do you find the median difference between two consecutive values?** 
    #### → (Hint: Lag diff + median)
40. **How do you get the “first non-null in each group by time”?** 
    #### → (Hint: Order + fetch first)

---

### **41–60: Subqueries, Filtering, Advanced Conditions**

41. **How do you return rows where the value is greater than the subquery’s maximum?** 
    #### → (Hint: Compare with scalar)
42. **How do you detect rows that match *all* filters from another table?** 
    #### → (Hint: Division logic)
43. **How do you filter based on a count from a correlated subquery?** 
    #### → (Hint: Correlated count)
44. **How do you find rows where a column matches the max of a filtered subset?** 
    #### → (Hint: Conditional max)
45. **How do you get “top X% per group”?** 
    #### → (Hint: Percentile-related rank)
46. **How do you detect when a row violates a uniqueness rule across multiple columns?** 
    #### → (Hint: Grouping sets)
47. **How do you detect when a value appears in exactly N different months?** 
    #### → (Hint: Count distinct month)
48. **How do you return only the first K rows per partition after a filter?** 
    #### → (Hint: Window filters)
49. **How do you find rows that fall inside the interquartile range?** 
    #### → (Hint: Q1–Q3)
50. **How do you return rows that match any of a dynamic set of values?** 
    #### → (Hint: Subquery IN)
51. **How do you detect rows where two values alternate (like 1,2,1,2)?** 
    #### → (Hint: Lag pattern)
52. **How do you find rows where a numeric value repeats more than X consecutive times?** 
    #### → (Hint: Streak detection)
53. **How do you detect when values change too frequently?** 
    #### → (Hint: Count changes)
54. **How do you return only rows that are not the min or max within their group?** 
    #### → (Hint: Filter extremes)
55. **How do you detect rows that violate a “must be increasing” rule within groups?** 
    #### → (Hint: Lag comparison)
56. **How do you find rows that satisfy two correlated conditions?** 
    #### → (Hint: Exists inside exists)
57. **How do you get the difference between a value and the group’s second highest value?** 
    #### → (Hint: Rank)
58. **How do you detect rows that are similar but not identical?** 
    #### → (Hint: Fuzzy match)
59. **How do you filter rows where a value appears more times than its group average count?** 
    #### → (Hint: Subquery counts)
60. **How do you return rows where a column’s value is within the top 5%?** 
    #### → (Hint: percentile_cont)

---

### **61–80: Strings, Text, Parsing, JSON, Regex**

61. **How do you find strings containing letters in strictly alphabetical order?** 
    #### → (Hint: Compare to sorted version)
62. **How do you extract the last number from a mixed string?** 
    #### → (Hint: Regex capture)
63. **How do you check if two strings differ in only one character?** 
    #### → (Hint: Compare positions)
64. **How do you find strings with repeated substrings like “ABABAB”?** 
    #### → (Hint: Regex repetition)
65. **How do you split a string using multiple delimiters?** 
    #### → (Hint: Regex split)
66. **How do you return the first word in a sentence?** 
    #### → (Hint: Locate space)
67. **How do you find strings that contain at least 3 digit sequences?** 
    #### → (Hint: Regex count)
68. **How do you mask part of a string except the first and last character?** 
    #### → (Hint: Substring + repeat)
69. **How do you detect strange whitespace like tabs inside a string?** 
    #### → (Hint: Regex whitespace classes)
70. **How do you find rows with invalid email formats?** 
    #### → (Hint: Regex pattern)
71. **How do you extract the domain from a list of URLs?** 
    #### → (Hint: Split on slashes)
72. **How do you compare JSON objects ignoring ordering?** 
    #### → (Hint: Normalize then compare)
73. **How do you check if JSON contains a specific nested value?** 
    #### → (Hint: JSON path query)
74. **How do you expand JSON arrays into separate rows?** 
    #### → (Hint: Cross apply)
75. **How do you flatten nested JSON into columns?** 
    #### → (Hint: Depth + apply)
76. **How do you store key–value pairs from JSON into relational tables?** 
    #### → (Hint: Shred JSON)
77. **How do you detect malformed XML?** 
    #### → (Hint: Try-parse)
78. **How do you combine string fields with NULLs without losing them?** 
    #### → (Hint: Concat_ws)
79. **How do you extract only letters from a noisy string?** 
    #### → (Hint: Regex replace)
80. **How do you validate that a phone number follows country rules?** 
    #### → (Hint: Regex with optional groups)

---

### **81–100: Performance, Optimization, Query Tuning**

81. **How do you detect a slow query caused by implicit conversions?** 
    #### → (Hint: Execution plan notes)
82. **How do you ensure a large join uses the index instead of scanning?** 
    #### → (Hint: Matching data types)
83. **How do you rewrite a slow correlated subquery into an efficient join?** 
    #### → (Hint: Pre-aggregate)
84. **How do you find top N groups efficiently on big tables?** 
    #### → (Hint: Partial sorting)
85. **How do you reduce hashing overhead in joins?** 
    #### → (Hint: Pre-sorting)
86. **How do you ensure your query uses partition pruning?** 
    #### → (Hint: Filter on partition column)
87. **How do you detect when a join is accidentally cross-joined?** 
    #### → (Hint: Row count explosion)
88. **How do you rewrite DISTINCT to run faster?** 
    #### → (Hint: Use group by)
89. **How do you reduce memory footprint in window-heavy queries?** 
    #### → (Hint: Subquery reuse)
90. **How do you identify bottlenecks in multi-join queries?** 
    #### → (Hint: Explain plan step-by-step)
91. **How do you ensure statistics are fresh for better optimization?** 
    #### → (Hint: Update stats)
92. **How do you diagnose parameter sniffing issues?** 
    #### → (Hint: Compare plans)
93. **How do you eliminate redundant joins for performance?** 
    #### → (Hint: Check unused columns)
94. **How do you accelerate full-text search conditions?** 
    #### → (Hint: Full-text index)
95. **How do you detect missing foreign key constraints causing bad plans?** 
    #### → (Hint: Referential clues)
96. **How do you tune complex expressions inside WHERE clauses?** 
    #### → (Hint: Precompute expressions)
97. **How do you detect spilled sorts in the execution plan?** 
    #### → (Hint: Spill to disk warnings)
98. **How do you rewrite LIKE queries for better performance?** 
    #### → (Hint: Avoid wildcard at front)
99. **How do you reduce overhead from huge IN (…) lists?** 
    #### → (Hint: Join with derived table)
100. **How do you verify if query parallelism actually helps?** 
    #### → (Hint: Compare cost models)

---
# 004. Climbing Stairs
# Application 1 (Simple): Direct application of the Fibonacci recurrence: DP[i]=DP[i−1]+DP[i−2] (1 or 2 steps allowed).

# recursive solution
def climbStairs_recursive(n: int) -> int:
    if n <= 2:
        return n
    return climbStairs_recursive(n - 1) + climbStairs_recursive(n - 2)

# recursive solution with memoization
def climbStairs_memoization(n: int, memo={}) -> int:  
    if n in memo:
        return memo[n]
    if n <= 2:
        return n
    memo[n] = climbStairs_memoization(n - 1, memo) + climbStairs_memoization(n - 2, memo)
    return memo[n]

# iterative solution with tabulation
def climbStairs_tabulation(n: int) -> int:
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

# iterative solution with space optimization
def climbStairs_space_optimized(n: int) -> int:   
    if n <= 2:
        return n
    first, second = 1, 2
    for i in range(3, n + 1):
        current = first + second
        first, second = second, current
    return second
# 005. Climbing Stairs with 3 Moves
# Application 2 (Extended): Direct application of the Tribonacci recurrence (1, 2, or 3 steps allowed).

# recursive solution
def climb_stairs_recursive(n):
    if n == 0:
        return 1
    if n < 0:
        return 0
    return (climb_stairs_recursive(n - 1) +
            climb_stairs_recursive(n - 2) +
            climb_stairs_recursive(n - 3)) 

# recursive solution with memoization
def climb_stairs_memoization(n, memo=None):
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n == 0:
        return 1
    if n < 0:
        return 0
    memo[n] = (climb_stairs_memoization(n - 1, memo) +
               climb_stairs_memoization(n - 2, memo) +
               climb_stairs_memoization(n - 3, memo))
    return memo[n]

# iterative solution with tabulation
def climb_stairs_tabulation(n):
    if n == 0:
        return 1
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        dp[i] = dp[i - 1]
        if i - 2 >= 0:
            dp[i] += dp[i - 2]
        if i - 3 >= 0:
            dp[i] += dp[i - 3]
    return dp[n]


# iterative solution with space optimization
def climb_stairs_space_optimized(n):
    if n == 0:
        return 1
    a, b, c = 1, 0, 0  # a = dp[i-1], b = dp[i-2], c = dp[i-3]
    for i in range(1, n + 1):
        current = a + b + c
        c = b
        b = a
        a = current
    return a
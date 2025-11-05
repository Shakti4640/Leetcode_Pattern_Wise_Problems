# 006. Weighted Climbing Stairs
# Variation 1 (Weighted): Extends the simple Climbing Stairs by adding a cost or weight to each step, but the goal is still counting ways.
# recursive solution without memoization
def climb_stairs_recursive(n, weights):
    if n == 0:
        return 1
    if n < 0:
        return 0
    return climb_stairs_recursive(n - 1, weights) + climb_stairs_recursive(n - 2, weights)

# recursive solution with memoization
def climb_stairs_memoization(n, weights, memo=None):
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n == 0:
        return 1
    if n < 0:
        return 0
    memo[n] = climb_stairs_memoization(n - 1, weights, memo) + climb_stairs_memoization(n - 2, weights, memo)
    return memo[n]

# iterative solution with tabulation
def climb_stairs_tabulation(n, weights):
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        dp[i] = dp[i - 1]
        if i - 2 >= 0:
            dp[i] += dp[i - 2]
    return dp[n]

# iterative solution with space optimization
def climb_stairs_space_optimized(n, weights):
    if n == 0:
        return 1
    first = 1  # dp[0]
    second = 0  # dp[-1]
    for i in range(1, n + 1):
        current = first
        if i - 2 >= 0:
            current += second
        second = first
        first = current
    return first
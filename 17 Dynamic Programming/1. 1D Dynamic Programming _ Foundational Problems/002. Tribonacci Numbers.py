# 002. Tribonacci Numbers
# Recurrence Extension: Extends the Fibonacci pattern to three terms: T(n)=T(n−1)+T(n−2)+T(n−3).

def tribonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1

    prev3 = 0  # T(0)
    prev2 = 1  # T(1)
    prev1 = 1  # T(2)

    for i in range(3, n + 1):
        current = prev1 + prev2 + prev3  # T(n) = T(n-1) + T(n-2) + T(n-3)
        prev3 = prev2
        prev2 = prev1
        prev1 = current

    return current  

# Example usage:
n = 10 # Change this value to compute a different Tribonacci number
print(f"Tribonacci number T({n}) is: {tribonacci(n)}")
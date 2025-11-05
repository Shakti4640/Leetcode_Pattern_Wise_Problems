# 001. Fibonacci Numbers
# Core Base Recurrence: The absolute simplest linear recurrence: F(n)=F(n−1)+F(n−2).

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1

    prev2 = 0  # F(0)
    prev1 = 1  # F(1)

    for i in range(2, n + 1):
        current = prev1 + prev2  # F(n) = F(n-1) + F(n-2)
        prev2 = prev1
        prev1 = current

    return current

# Example usage:
n = 10 # Change this value to compute a different Fibonacci number
print(f"Fibonacci number F({n}) is: {fibonacci(n)}")

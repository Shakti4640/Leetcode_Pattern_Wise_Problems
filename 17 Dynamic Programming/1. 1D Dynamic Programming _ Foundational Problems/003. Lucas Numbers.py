# 003. Lucas Numbers
# Recurrence Reinforcement: Uses the same recurrence as Fibonacci, but with different initial base cases.

def lucas_number(n: int) -> int:
    if n == 0:
        return 2
    elif n == 1:
        return 1
    
    prev2, prev1 = 2, 1
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return current

# Example usage:
n = 5 # Change this value to test with different inputs
print(f"The {n}th Lucas number is: {lucas_number(n)}")
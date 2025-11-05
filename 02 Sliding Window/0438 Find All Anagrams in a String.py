# 438 Find All Anagrams in a String
# Sliding window with frequency counters, check when matches target

def findAnagrams(s, p):
    if not s or not p or len(p) > len(s):
        return []
    
    from collections import Counter
    
    p_count = Counter(p)
    s_count = Counter()
    result = []
    
    for i in range(len(s)):
        s_count[s[i]] += 1
        
        # Remove the leftmost character when the window size exceeds p's length
        if i >= len(p):
            if s_count[s[i - len(p)]] == 1:
                del s_count[s[i - len(p)]]
            else:
                s_count[s[i - len(p)]] -= 1
        
        # Compare counts when we have a full window
        if s_count == p_count:
            result.append(i - len(p) + 1)
    
    return result
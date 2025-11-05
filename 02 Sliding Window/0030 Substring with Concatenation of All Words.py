# 30 Substring with Concatenation of All Words
# Sliding window with word frequency checks

def findSubstring(s, words):
    if not s or not words:
        return []

    word_length = len(words[0])
    word_count = len(words)
    total_length = word_length * word_count
    word_map = {}
    
    for word in words:
        if word in word_map:
            word_map[word] += 1
        else:
            word_map[word] = 1

    result = []
    
    for i in range(len(s) - total_length + 1):
        seen_words = {}
        j = 0
        
        while j < word_count:
            start_index = i + j * word_length
            word = s[start_index:start_index + word_length]
            
            if word not in word_map:
                break
            
            if word in seen_words:
                seen_words[word] += 1
            else:
                seen_words[word] = 1
            
            if seen_words[word] > word_map[word]:
                break
            
            j += 1
        
        if j == word_count:
            result.append(i)
    
    return result
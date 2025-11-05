# 524 Longest Word in Dictionary through Deleting
# Check subsequence for each dictionary word, track longest

class Solution:
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        def is_subsequence(word, s):
            # Two pointers to check if word is a subsequence of s
            i = 0
            for char in s:
                if i < len(word) and word[i] == char:
                    i += 1
            return i == len(word)
        
        # Sort: first by length descending, then lexicographically ascending
        dictionary.sort(key=lambda x: (-len(x), x))

        for word in dictionary:
            if is_subsequence(word, s):
                return word

        return ""

# **Python Programming Interview Questions**

---

### **Batch 1: Python Basics & Core Programming (Q1–Q100)**

#### **Core Syntax & Variables**

1. What is Python, and how is it different from other programming languages?
   → Python is a beginner-friendly language that focuses on readability and simplicity compared to many others.

2. How do you declare a variable in Python?
   → Just assign a value like `x = 5`—no type declaration needed.

3. What are Python’s standard data types?
   → Numbers, strings, lists, tuples, dictionaries, sets, and booleans.

4. What is the difference between mutable and immutable types?
   → Mutable types can change their contents, immutable ones cannot.

5. Explain the difference between `is` and `==`.
   → `==` checks value equality, while `is` checks if two things are the exact same object.

6. What are Python’s basic arithmetic operators?
   → `+`, `-`, `*`, `/`, `//`, `%`, and `**`.

7. How does Python handle operator precedence?
   → It follows a fixed priority order (like maths) to decide which operation runs first.

8. What is the difference between `//` and `/`?
   → `/` gives a float result, while `//` gives a floor (rounded-down) result.

9. How do you swap two variables without a temporary variable?
   → Use `a, b = b, a`.

10. Explain the difference between `str()`, `repr()`, and `format()`.
    → `str()` makes things user-friendly, `repr()` makes them unambiguous for developers, and `format()` lets you style the output.

#### **Control Flow**

11. Explain the syntax of an `if-elif-else` statement.
    → It checks conditions in order using `if`, then `elif` if needed, and `else` when nothing else matches.

12. What is the difference between `while` and `for` loops?
    → `while` repeats until a condition changes, while `for` repeats over items in a sequence.

13. How do `break`, `continue`, and `pass` statements work?
    → `break` stops the loop, `continue` skips to the next round, and `pass` does nothing at all.

14. How do you loop over a dictionary?
    → You loop using `for key, value in dict.items()`.

15. How do you loop over a list with indexes?
    → Use `for i in range(len(my_list))`.

16. Explain Python’s `enumerate()` function.
    → It gives you both the index and the value while looping.

17. What is the purpose of the `else` clause in loops?
    → It runs only if the loop finishes normally without a `break`.

18. How do nested loops work in Python?
    → One loop runs inside another so the inner loop repeats fully for each outer loop step.

19. Explain the difference between `range()` and `xrange()` (Python 2 vs 3).
    → `range()` makes a list in Python 2 but works like `xrange()` in Python 3, which generates values on the fly.

20. How would you iterate over two lists simultaneously?
    → Use `for a, b in zip(list1, list2)`.

#### **Functions**

21. How do you define a function in Python?
    → You create one with `def name():` like a tiny machine that runs when called.

22. Explain positional vs keyword arguments.
    → Positional go by order, keyword go by name—like calling someone by seat number vs calling them by name.

23. What are default arguments in Python functions?
    → They’re backup values the function uses if you don’t give your own.

24. How do you define a function with variable arguments (`*args`, `**kwargs`)?
    → Use `*args` for extra unnamed stuff and `**kwargs` for extra named stuff like a magical backpack of inputs.

25. What is a lambda function?
    → A teeny-tiny one-line function used when you’re in a hurry.

26. How do you return multiple values from a function?
    → Just separate them with commas and they come back as a tuple.

27. What is recursion, and give an example?
    → It’s when a function calls itself, like a mirror reflecting another mirror.

28. Explain the difference between a function and a method.
    → A method is just a function that lives inside an object and behaves politely about it.

29. What are docstrings, and how are they used?
    → They’re little notes inside triple quotes that explain what your function does.

30. How do you import a function from another module?
    → Use `from module import function` like borrowing a tool from a neighbor.

#### **Data Structures: Lists**

31. How do you create a list in Python?
    → Put values inside brackets like `[1, 2, 3]`.

32. How do you access elements in a list?
    → Use indexes like `my_list[0]`.

33. How do you slice a list?
    → Use `my_list[start:end]` to grab a piece of it.

34. How do you add an element to a list?
    → Use `append()` or `insert()` depending on where you want it.

35. How do you remove an element from a list?
    → Use `remove()`, `pop()`, or `del` depending on your mood.

36. How do you sort a list?
    → Use `sort()` for in-place or `sorted()` for a fresh sorted copy.

37. How do you reverse a list?
    → Use `reverse()` or slice it like `my_list[::-1]`.

38. What are list comprehensions, and give an example?
    → They’re quick list-makers like `[x*2 for x in nums]`.

39. How do you check if an element exists in a list?
    → Use `if item in my_list`.

40. How do you copy a list (shallow vs deep copy)?
    → Shallow uses `list.copy()`; deep uses `copy.deepcopy()` for fully separate copies.

#### **Data Structures: Tuples**

41. What is a tuple in Python?
    → It’s an ordered collection that you can’t change once created.

42. How is a tuple different from a list?
    → A list can change, a tuple stays frozen.

43. How do you access elements in a tuple?
    → Use indexing like `t[0]`.

44. Can you modify a tuple? Explain.
    → Nope, it’s immutable so its contents are locked in.

45. What are tuple packing and unpacking?
    → Packing bundles values into a tuple, unpacking splits them back out into variables.

46. How can tuples be used as dictionary keys?
    → Because tuples never change, they work as stable keys.

47. Explain nested tuples with an example.
    → A tuple inside another like `(1, (2, 3))`.

48. How do you iterate over a tuple?
    → Use a simple `for item in tuple:` loop.

49. What is the purpose of the `namedtuple` in `collections`?
    → It gives tuples names for their fields so they behave like lightweight objects.

50. How do you convert a tuple to a list and vice versa?
    → Use `list()` to turn a tuple into a list and `tuple()` to turn a list into a tuple.

#### **Data Structures: Dictionaries**

51. How do you create a dictionary?
    → Use curly braces like `{"a": 1, "b": 2}`.

52. How do you access values in a dictionary?
    → Use the key like `my_dict["a"]`.

53. How do you add or update dictionary elements?
    → Assign with `my_dict[key] = value`.

54. How do you remove elements from a dictionary?
    → Use `pop()`, `del`, or `popitem()`.

55. How do you iterate over keys, values, and items?
    → Use `dict.keys()`, `dict.values()`, and `dict.items()`.

56. What is the purpose of `defaultdict`?
    → It gives automatic default values so you don’t get key errors.

57. How do you merge two dictionaries?
    → Use `{**d1, **d2}` or `d1.update(d2)`.

58. How do you sort a dictionary by keys or values?
    → Use `sorted()` on `dict.items()` with a key function.

59. Explain dictionary comprehension.
    → It’s a quick way to build dictionaries like `{x: x*2 for x in nums}`.

60. Can dictionary keys be mutable? Explain.
    → No, keys must stay the same forever, so only immutable types qualify.

#### **Data Structures: Sets**

61. How do you create a set in Python?
    → Use curly braces like `{1, 2, 3}` or `set()`.

62. What are the main properties of sets?
    → They’re unordered, unique, and super fast for membership checks.

63. How do you add or remove elements in a set?
    → Use `add()` to include and `remove()` or `discard()` to take away.

64. What is the difference between `remove()` and `discard()`?
    → `remove()` complains if the item’s missing, `discard()` stays chill.

65. How do you perform union, intersection, and difference on sets?
    → Use `|`, `&`, and `-` or their method versions.

66. How do you check if a set is a subset or superset of another set?
    → Use `<`, `<=`, `>`, `>=` or the `.issubset()` and `.issuperset()` methods.

67. What is a frozenset?
    → A set that can’t be changed, like a set in winter mode.

68. How do you iterate over a set?
    → Use a simple `for item in my_set:` loop.

69. Can sets contain duplicate elements?
    → No, they automatically toss out repeats.

70. Explain practical use cases of sets.
    → They’re perfect for removing duplicates, fast lookups, and comparing groups of things.

#### **Exception Handling**

71. What is exception handling in Python?
    → It’s the way Python deals with errors without crashing your program.

72. Explain `try`, `except`, `finally`, and `else` blocks.
    → `try` tests code, `except` handles errors, `else` runs if no error happens, and `finally` always runs at the end.

73. How do you raise an exception in Python?
    → Use `raise SomeError("message")`.

74. What are some common Python exceptions?
    → `ValueError`, `TypeError`, `KeyError`, `IndexError`, `ZeroDivisionError`.

75. How do you create a custom exception class?
    → Make a class that inherits from `Exception`.

76. What is the difference between `Exception` and `BaseException`?
    → `BaseException` is the root of all errors, and `Exception` is the normal branch most errors use.

77. How do you catch multiple exceptions in a single block?
    → Put them in a tuple like `except (TypeError, ValueError):`.

78. What is the use of `with` statement for exception handling?
    → It safely manages resources so they close even if errors happen.

79. Explain the difference between syntax errors and runtime errors.
    → Syntax errors break the code before it runs; runtime errors happen while running.

80. How do you log exceptions for debugging?
    → Use the `logging` module and `logging.exception()` to record error details.

#### **File I/O**

81. How do you open and close a file in Python?
    → Use `open()` to open and `close()` to shut it.

82. What is the difference between text and binary files?
    → Text stores characters, binary stores raw bytes.

83. How do you read a file line by line?
    → Loop over the file object directly.

84. How do you write to a file?
    → Open it in write mode and use `write()`.

85. What is the difference between `read()`, `readline()`, and `readlines()`?
    → `read()` gets everything, `readline()` gets one line, `readlines()` gets all lines in a list.

86. How do you append to a file?
    → Open it in `'a'` mode and use `write()`.

87. What are file modes in Python (`r`, `w`, `a`, `b`)?
    → `r` reads, `w` overwrites, `a` adds, `b` switches to binary.

88. How do you use a context manager (`with`) for file handling?
    → Wrap the file in `with` so it closes itself automatically.

89. How do you check if a file exists before opening it?
    → Use `os.path.exists()`.

90. How do you handle file exceptions safely?
    → Wrap file operations in `try-except` blocks.

#### **Modules, Packages & Virtual Environments**

91. What is a module in Python?
    → It’s just a file full of Python code you can reuse.

92. How do you import a module?
    → Use `import module_name`.

93. Explain the difference between `import module` and `from module import function`.
    → One brings the whole toolbox, the other grabs just one tool.

94. How do you create a package in Python?
    → Put modules in a folder with an `__init__.py` file.

95. What is the purpose of `__init__.py`?
    → It tells Python, “Hey, this folder is a package!”

96. How do you list installed packages?
    → Run `pip list`.

97. How do you install a package using `pip`?
    → Use `pip install package_name`.

98. How do you create a virtual environment?
    → Run `python -m venv envname`.

99. Why are virtual environments important?
    → They keep your project’s packages nicely separated from others.

100. How do you activate and deactivate a virtual environment?
     → Activate with `source env/bin/activate` (or `Scripts\activate` on Windows) and deactivate with `deactivate`.

---

### **Batch 2: Advanced Python Programming (Q101–Q200)**

#### **Iterators & Generators**

101. What is an iterator in Python?
     → It’s something you can move through one item at a time.

102. How does the `iter()` function work?
     → It turns an iterable into an iterator.

103. Explain the `next()` function.
     → It gives you the next item from an iterator.

104. How do you create a custom iterator class?
     → Define `__iter__()` and `__next__()` inside a class.

105. What is a generator in Python?
     → It’s a special function that makes an iterator automatically.

106. How does `yield` differ from `return`?
     → `yield` pauses the function, `return` stops it for good.

107. What are generator expressions?
     → They’re tiny generators written like `(x*2 for x in nums)`.

108. How do generators help with memory efficiency?
     → They produce items only when needed instead of storing everything.

109. Explain the difference between an iterator and an iterable.
     → An iterable can give you an iterator; an iterator actually moves through items.

110. How do you chain generators for complex pipelines?
     → You stack them by feeding one generator’s output into the next.

#### **Comprehensions**

111. Explain list comprehensions with an example.
     → They build lists fast, like `[x*x for x in nums]`.

112. How do dictionary comprehensions work?
     → They create key–value pairs quickly like `{x: x*x for x in nums}`.

113. Explain set comprehensions.
     → They make sets using a similar style like `{x*x for x in nums}`.

114. How do you use nested comprehensions?
     → Put one inside another like `[a+b for a in A for b in B]`.

115. How can you add conditions to comprehensions?
     → Add an `if` like `[x for x in nums if x%2==0]`.

116. What is the difference between a comprehension and a generator expression?
     → One builds the whole structure; the other makes items on demand.

117. How do comprehensions improve code readability?
     → They shrink big loops into neat one-liners.

118. Can comprehensions replace loops entirely?
     → Not always, but they handle many loop scenarios nicely.

119. Explain the performance benefits of comprehensions.
     → They run faster because they’re optimized inside Python.

120. How do you handle exceptions in comprehensions?
     → Wrap the whole comprehension in a `try-except` since you can’t catch inside it cleanly.

#### **Object-Oriented Programming (OOP)**

121. What is a class in Python?
     → It’s a blueprint for creating objects.

122. What is an object?
     → It’s a thing made from a class with its own data and abilities.

123. Explain the difference between instance, class, and static methods.
     → Instance methods use `self`, class methods use `cls`, and static methods use neither.

124. What is inheritance? Give an example.
     → It lets one class get features from another, like `Dog(Animal)`.

125. What is multiple inheritance, and how does Python handle it?
     → A class can inherit from many parents, and Python uses MRO to decide who gets priority.

126. What is polymorphism in Python?
     → Different objects can use the same method name but behave differently.

127. Explain encapsulation and how it is implemented.
     → It hides internal details using naming conventions like `_name`.

128. What are private, protected, and public attributes in Python?
     → Public for everyone, protected with one underscore, private with two underscores.

129. What is method overriding?
     → A child class replaces a parent class’s method with its own version.

130. Explain method overloading in Python.
     → Python fakes it using default or variable arguments since it doesn’t support it directly.

#### **Magic Methods & Special Functions**

131. What are magic methods in Python?
     → Special methods with double underscores that let objects behave in magical ways.

132. Explain `__init__` and `__new__`.
     → `__new__` makes the object, `__init__` sets it up.

133. How do `__str__` and `__repr__` differ?
     → `__str__` is friendly, `__repr__` is technical.

134. What are `__call__` and `__getitem__` used for?
     → `__call__` lets an object act like a function, `__getitem__` handles indexing.

135. Explain `__len__`, `__iter__`, and `__next__`.
     → They let objects act like sequences you can measure and loop through.

136. How do `__enter__` and `__exit__` relate to context managers?
     → They power the `with` statement by setting up and cleaning up.

137. Explain operator overloading with `__add__` or `__mul__`.
     → These methods let your objects respond to `+` and `*` in custom ways.

138. What is `__dict__` used for?
     → It stores an object’s attributes in a handy dictionary.

139. How do you implement custom comparison operators?
     → Define methods like `__lt__`, `__gt__`, `__eq__`, and friends.

140. How can magic methods enhance code readability?
     → They make objects act naturally so your code feels smooth and intuitive.

#### **Properties, Descriptors & Metaclasses**

141. What is the `property()` function used for?
     → It turns methods into managed attributes with built-in getters and setters.

142. How do you define getter, setter, and deleter in Python?
     → Use `@property`, `@attr.setter`, and `@attr.deleter` decorators.

143. What is a descriptor?
     → It’s an object that controls how attributes are accessed.

144. How do you implement custom descriptors?
     → Define a class with `__get__`, `__set__`, or `__delete__`.

145. What are the use cases for descriptors?
     → They’re great for validation, computed attributes, and shared behavior.

146. What is a metaclass in Python?
     → It’s a class that creates other classes.

147. How do you create a custom metaclass?
     → Inherit from `type` and override methods like `__new__` or `__init__`.

148. Explain the role of `type()` in metaclasses.
     → `type()` is the default metaclass that builds classes.

149. How do metaclasses control class creation?
     → They intercept the class-building process and can modify it.

150. Give a practical example of using metaclasses.
     → They can auto-add logging, enforce naming rules, or register classes automatically.

#### **Functional Programming**

151. What is functional programming?
     → It’s a style where you build programs using functions without changing data.

152. Explain the use of `map()` with an example.
     → `map(f, items)` applies a function to each item, like `map(str.upper, words)`.

153. How does `filter()` work in Python?
     → It keeps only the items that pass a test function.

154. Explain `reduce()` and where it is used.
     → It combines items into one value, like adding everything up.

155. What is `functools` in Python?
     → A module full of handy functional tools.

156. Explain `partial()` from `functools`.
     → It pre-fills part of a function’s arguments to make a new function.

157. How do higher-order functions work in Python?
     → They accept functions as inputs or return them as outputs.

158. What are pure functions?
     → Functions that don’t change anything and always give the same output for the same input.

159. Explain immutability in functional programming.
     → It means data never changes, so you make new copies instead.

160. How does Python support first-class functions?
     → Functions can be stored, passed, returned, and treated like any other value.

#### **Decorators**

161. What is a decorator in Python?
     → It’s a function that adds extra superpowers to another function.

162. How do you create a simple decorator function?
     → Make a wrapper inside a function and return that wrapper.

163. Explain the difference between function decorators and class decorators.
     → One decorates functions, the other decorates entire classes.

164. How do you use `functools.wraps`?
     → Put it on your wrapper so the original function’s name and docs stay intact.

165. Can decorators accept arguments?
     → Yep—just wrap another layer around the decorator.

166. Explain chaining decorators.
     → Stack them so a function passes through several decorators.

167. How can decorators improve code reusability?
     → They let you reuse behavior without rewriting it everywhere.

168. Give a practical example of a logging decorator.
     → A decorator that prints function name and arguments before running it.

169. How do property decorators work?
     → They turn methods into attribute-like access using `@property`.

170. Explain the difference between `@staticmethod`, `@classmethod`, and `@property`.
     → Static ignores the class, classmethod uses the class, property acts like a managed attribute.

#### **Recursion**

171. What is recursion, and when should it be used?
     → It’s when a function calls itself, useful for problems that split into smaller versions of themselves.

172. Explain base case and recursive case.
     → The base case stops the recursion; the recursive case keeps it going.

173. How do you prevent infinite recursion?
     → Always include a base case that actually gets reached.

174. Compare recursion with iteration.
     → Recursion uses repeated self-calls, iteration uses loops.

175. How do you calculate factorial using recursion?
     → Define `n * factorial(n-1)` with a base case of `1`.

176. How do you compute Fibonacci numbers using recursion?
     → Return `fib(n-1) + fib(n-2)` with base cases `0` and `1`.

177. What is tail recursion?
     → A recursive call made as the final action in a function.

178. Does Python optimize tail recursion?
     → Nope, Python leaves it unoptimized.

179. How can recursion affect memory (stack overflow)?
     → Too many nested calls fill up the call stack.

180. Give a real-world example of recursion.
     → A folder containing folders inside folders.

#### **Memory Management**

181. How does Python manage memory internally?
     → It uses private heaps, reference counting, and a garbage collector.

182. What is reference counting?
     → A counter tracks how many things point to an object.

183. How does Python’s garbage collector work?
     → It cleans up leftover objects, especially those in cycles.

184. What are circular references, and how are they handled?
     → Objects that reference each other, handled by the cycle detector.

185. Explain `__del__` method in Python.
     → It runs when an object is about to be destroyed.

186. How do `weakref` objects work?
     → They let you reference objects without increasing their reference count.

187. How do you manually trigger garbage collection?
     → Call `gc.collect()`.

188. What are memory leaks in Python?
     → When objects never get freed because something still references them.

189. How can you optimize memory usage in Python programs?
     → Use generators, avoid unnecessary copies, and prefer lightweight structures.

190. Explain the difference between shallow and deep copies.
     → Shallow copies copy references; deep copies clone everything fully.

#### **Advanced Python Concepts**

191. What is duck typing in Python?
     → If something behaves like the right type, Python treats it as that type.

192. Explain EAFP vs LBYL programming styles.
     → EAFP tries first and handles errors; LBYL checks everything before acting.

193. What are Python’s built-in functions for introspection?
     → `type()`, `id()`, `dir()`, `hasattr()`, `getattr()`, and friends.

194. How do you dynamically import a module?
     → Use `importlib.import_module()`.

195. How do you check if an object has an attribute?
     → Use `hasattr(obj, "name")`.

196. What is the Global Interpreter Lock (GIL)?
     → A lock that allows only one thread to run Python code at a time.

197. How does GIL affect multithreading in Python?
     → CPU-heavy threads can’t run in true parallel.

198. How do you bypass GIL limitations?
     → Use multiprocessing or extension modules that release the GIL.

199. Explain the difference between multithreading and multiprocessing.
     → Threads share memory; processes run separately for real parallelism.

200. How can Python’s `ctypes` or `cffi` improve performance?
     → They let you call fast C code directly from Python.

---

### **Batch 3: Python Standard Libraries & Tools (Q201–Q300)**

#### **Built-in Libraries**

201. What is the purpose of the `os` module?
     → It lets Python talk to your operating system.

202. How do you list all files in a directory using `os`?
     → Use `os.listdir(path)`.

203. How do you create, remove, and rename directories in Python?
     → Use `os.mkdir()`, `os.rmdir()`, and `os.rename()`.

204. What is `os.path` used for?
     → Handling file paths safely and easily.

205. How do you get the absolute path of a file?
     → Use `os.path.abspath(filename)`.

206. Explain `os.environ` and how to access environment variables.
     → It’s a dictionary of system variables you read like `os.environ["KEY"]`.

207. What is the `sys` module used for?
     → It gives Python insight into the interpreter and system details.

208. How do you get command-line arguments using `sys.argv`?
     → Read them from the `sys.argv` list.

209. How can you exit a Python program using `sys`?
     → Call `sys.exit()`.

210. How do you check Python version using `sys`?
     → Look at `sys.version` or `sys.version_info`.

#### **Date and Time**

211. How do you get the current date and time using `datetime`?
     → Use `datetime.datetime.now()`.

212. How do you format a datetime object as a string?
     → Use `strftime()` with a format pattern.

213. How do you parse a string into a datetime object?
     → Use `strptime()` with the matching format.

214. Explain the difference between `datetime`, `date`, and `time` objects.
     → `date` stores only the day, `time` only the clock, `datetime` stores both.

215. How do you calculate the difference between two dates?
     → Subtract them to get a `timedelta`.

216. What is `timedelta` used for?
     → Representing time differences and doing date arithmetic.

217. How do you get the current timestamp?
     → Use `time.time()` or `datetime.now().timestamp()`.

218. How do you convert between timestamp and datetime?
     → Use `datetime.fromtimestamp()` and `.timestamp()`.

219. Explain time zones handling in Python.
     → Use `datetime` with `tzinfo` or the `zoneinfo` module.

220. How do you measure execution time of code snippets using `time` module?
     → Capture `time.time()` before and after, then subtract.

#### **Mathematical Operations**

221. What is the `math` module used for?
     → It provides fast, accurate math functions.

222. How do you calculate factorial and power using `math`?
     → Use `math.factorial()` and `math.pow()`.

223. Explain trigonometric functions in the `math` module.
     → They include `sin`, `cos`, `tan`, and their inverses.

224. How do you generate random numbers using `random` module?
     → Use functions like `random.random()` or `random.randint()`.

225. How do you select a random element from a list?
     → Use `random.choice(list)`.

226. How do you shuffle a list randomly?
     → Use `random.shuffle(list)`.

227. How do you generate random numbers within a range?
     → Use `random.randint(a, b)` or `randrange()`.

228. Explain probability distributions in the `random` module.
     → It includes tools like `random.uniform()`, `random.gauss()`, and others.

229. How do you calculate the greatest common divisor (GCD)?
     → Use `math.gcd(a, b)`.

230. How do you compute square roots and logarithms using `math`?
     → Use `math.sqrt()` and `math.log()`.

#### **Text Processing & Regular Expressions**

231. What is the `re` module used for?
     → It handles pattern matching with regular expressions.

232. How do you search for a pattern in a string?
     → Use `re.search(pattern, text)`.

233. How do you match a string exactly?
     → Use `re.fullmatch(pattern, text)`.

234. How do you extract all matches from a string?
     → Use `re.findall(pattern, text)`.

235. How do you replace text using regex?
     → Use `re.sub(pattern, replacement, text)`.

236. Explain regex groups and capturing groups.
     → Groups capture parts of the match using parentheses.

237. How do you split a string using a regex pattern?
     → Use `re.split(pattern, text)`.

238. What are raw strings (`r"pattern"`) used for in regex?
     → They prevent Python from treating backslashes specially.

239. How do you perform case-insensitive regex matching?
     → Add the `re.IGNORECASE` flag.

240. How do you compile a regex pattern for repeated use?
     → Use `re.compile(pattern)` to reuse it efficiently.

#### **String Handling**

241. How do you convert a string to uppercase or lowercase?
     → Use `.upper()` or `.lower()`—like giving the string a mood change.

242. How do you strip whitespace from a string?
     → Use `.strip()` to sweep away the extra spaces.

243. How do you split a string into a list?
     → Use `.split()` to break it apart like cracking a cookie.

244. How do you join a list of strings into a single string?
     → Use `'separator'.join(list)` to glue them back together.

245. Explain string formatting using `%`, `str.format()`, and f-strings.
     → `%` is old-school, `.format()` is polite, f-strings are the cool modern shortcut.

246. How do you check if a string starts or ends with a substring?
     → Use `.startswith()` or `.endswith()` like checking the front door or back door.

247. How do you find the index of a substring?
     → Use `.find()` to locate it without throwing a tantrum.

248. How do you replace substrings in a string?
     → Use `.replace()` to swap the old with the new.

249. How do you check if a string contains only digits or letters?
     → Use `.isdigit()` or `.isalpha()`—like giving the string a quick identity check.

250. How do you handle Unicode strings in Python?
     → Just use normal strings; Python 3 treats them all as Unicode by default.

#### **Collections**

251. What is the `collections` module used for?
     → It offers special container types that act like upgraded data-structure tools.

252. Explain the `deque` data structure.
     → It’s a double-ended queue that lets you add or remove items super fast from both ends.

253. How do you append and pop elements from both ends of a `deque`?
     → Use `append()`, `appendleft()`, `pop()`, and `popleft()`—like doors on both sides.

254. What is a `Counter`, and how is it used?
     → It’s a tally machine that counts how many times each item appears.

255. How do you count elements in a list using `Counter`?
     → Just do `Counter(your_list)` and it counts everything for you.

256. What is a `defaultdict`, and how is it different from a normal dict?
     → It auto-creates default values instead of complaining about missing keys.

257. How do you provide a default factory function for `defaultdict`?
     → Pass a function like `int` or `list` when creating it: `defaultdict(list)`.

258. Explain `OrderedDict` and its use cases.
     → It remembers the order you insert items—handy when order matters.

259. How is an `OrderedDict` different from a regular dictionary?
     → Older Python relied on it for order; now it mainly adds extra ordering tricks.

260. What are named tuples, and when would you use them?
     → They’re lightweight, tuple-like objects with named fields—great for clean, readable data.

#### **Logging**

261. What is the `logging` module used for?
     → It lets you record what your program is doing, like keeping a diary for debugging.

262. How do you create a basic logger?
     → Use `logging.basicConfig()` and then call the logging functions.

263. Explain logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL).
     → They’re message seriousness levels, from tiny whispers to full-blown alarms.

264. How do you log messages to a file?
     → Use `basicConfig(filename='file.log')` to send logs straight into a file.

265. How do you format log messages?
     → Pass `format='...'` in `basicConfig()` to style your log messages.

266. How do you create multiple loggers in a program?
     → Use `logging.getLogger(name)` for each unique logger you want.

267. What is the difference between `logging.debug()` and `logger.debug()`?
     → The first uses the root logger; the second uses your custom logger.

268. How do you disable logging in production code?
     → Call `logging.disable(logging.CRITICAL)` to silence everything.

269. How do you rotate log files automatically?
     → Use `RotatingFileHandler` or `TimedRotatingFileHandler` to auto-manage file sizes or time.

270. What is the difference between `StreamHandler` and `FileHandler`?
     → One sends logs to the screen, the other writes them into a file notebook.

#### **Command-Line Argument Parsing**

271. What is the `argparse` module used for?
     → It helps you build user-friendly command-line interfaces.

272. How do you define command-line arguments?
     → Create a parser with `ArgumentParser()` and add arguments using `add_argument()`.

273. How do you specify default values for arguments?
     → Use the `default=` parameter in `add_argument()`.

274. How do you make an argument required?
     → Set `required=True` in `add_argument()`.

275. How do you parse arguments from `sys.argv`?
     → Call `parser.parse_args()` and it reads them automatically.

276. How do you add help messages for arguments?
     → Use the `help=` parameter in `add_argument()`.

277. How do you handle optional flags in `argparse`?
     → Use arguments starting with `--` and set `action='store_true'` or similar.

278. How do you group arguments into mutually exclusive sets?
     → Use `parser.add_mutually_exclusive_group()`.

279. How do you access parsed arguments?
     → Use dot notation like `args.name` after calling `parse_args()`.

280. How do you handle subcommands in `argparse`?
     → Create subparsers using `parser.add_subparsers()` and add commands to them.

#### **Configuration Handling**

281. What is the `configparser` module used for?
     → It reads and writes INI-style configuration files.

282. How do you read an INI file using `configparser`?
     → Create a `ConfigParser()` and call `read('file.ini')`.

283. How do you write to an INI file?
     → Modify the parser and save using `write()` on a file handle.

284. How do you add sections and options programmatically?
     → Use `add_section()` and then set values like a dictionary.

285. How do you access configuration values safely?
     → Use `.get()` to avoid errors when keys are missing.

286. How do you set default values in `configparser`?
     → Pass a `defaults=` dictionary when creating the parser.

287. Can `configparser` handle data types other than strings?
     → Yes, using `.getint()`, `.getfloat()`, or `.getboolean()`.

288. How do you remove sections and options?
     → Use `remove_section()` or `remove_option()`.

289. How do you interpolate values from other sections?
     → Enable interpolation with `ConfigParser(interpolation=...)`.

290. What are the alternatives to `configparser` for configuration management?
     → JSON, YAML, TOML, environment variables, or custom classes.

#### **Unit Testing**

291. What is unit testing, and why is it important?
     → It checks small pieces of code alone to catch mistakes early.

292. What is the `unittest` module?
     → It’s Python’s built-in framework for writing and running tests.

293. How do you define a test case using `unittest`?
     → Make a class inheriting from `unittest.TestCase` and add methods starting with `test_`.

294. How do you assert that two values are equal?
     → Use `self.assertEqual(a, b)`.

295. How do you assert that an exception is raised?
     → Use `with self.assertRaises(ExceptionType):`.

296. How do you run multiple test cases together?
     → Put them in the same file or directory and run the test runner.

297. What is `setUp()` and `tearDown()` in `unittest`?
     → They prepare things before each test and clean up afterward.

298. How do you skip a test case conditionally?
     → Use `@unittest.skipIf(condition, reason)`.

299. How do you use `pytest` as an alternative to `unittest`?
     → Install it, write plain functions starting with `test_`, and run `pytest`.

300. How do you mock external dependencies in tests?
     → Use `unittest.mock` tools like `patch()` to replace real objects with fakes.

---


### **Batch 4: NumPy Fundamentals (Q301–Q400)**

#### **Array Creation**

301. What is NumPy, and why is it used?
     → It’s a fast math library that handles big arrays like a pro.

302. How do you import NumPy in Python?
     → Use `import numpy as np`.

303. How do you create a 1D array using `np.array()`?
     → Pass a list: `np.array([1, 2, 3])`.

304. How do you create a 2D or 3D array?
     → Use nested lists inside `np.array()`.

305. How do you create an array of zeros?
     → Use `np.zeros(shape)`.

306. How do you create an array of ones?
     → Use `np.ones(shape)`.

307. How do you create an identity matrix?
     → Use `np.eye(n)`.

308. How do you create an array with a range of numbers?
     → Use `np.arange(start, stop, step)`.

309. How do you create a linearly spaced array using `linspace()`?
     → Call `np.linspace(start, stop, count)`.

310. How do you create a random array using `np.random`?
     → Use functions like `np.random.rand()` or `np.random.randn()`.

#### **Array Indexing & Slicing**

311. How do you access a single element in a NumPy array?
     → Use square brackets like `arr[i]`.

312. How do you access a row or column in a 2D array?
     → Row: `arr[i]`, Column: `arr[:, j]`.

313. How do you slice a 1D array?
     → Use `arr[start:end]` like slicing a loaf of bread.

314. How do you slice a 2D array?
     → Use `arr[r1:r2, c1:c2]` to grab a block.

315. What is fancy indexing in NumPy?
     → Picking elements using lists of indices instead of single numbers.

316. How do you use boolean indexing to filter arrays?
     → Use a condition like `arr[arr > 5]`.

317. How do you assign values to specific indices using indexing?
     → Just index and assign: `arr[i] = value`.

318. How do you access elements using negative indices?
     → Use `arr[-1]` etc. to reach from the end backward.

319. How do you reshape an array without changing its data?
     → Use `arr.reshape(new_shape)`.

320. How do you flatten a multidimensional array?
     → Use `arr.ravel()` or `arr.flatten()`.

#### **Array Operations**

321. How do you perform element-wise addition, subtraction, multiplication, and division?
     → Just use `+`, `-`, `*`, `/` directly on arrays.

322. How do you compute the sum of all elements in an array?
     → Call `arr.sum()`.

323. How do you compute the mean, median, and standard deviation?
     → Use `arr.mean()`, `np.median(arr)`, and `arr.std()`.

324. How do you find the maximum and minimum values in an array?
     → Use `arr.max()` and `arr.min()`.

325. How do you perform cumulative sum and cumulative product?
     → Use `arr.cumsum()` and `arr.cumprod()`.

326. How do you compute dot product and matrix multiplication?
     → Use `np.dot(a, b)` or `a @ b`.

327. How do you perform broadcasting in NumPy?
     → Let NumPy stretch smaller arrays to match shapes automatically.

328. What is the difference between `*` and `np.dot()` for arrays?
     → `*` is element-wise; `dot()` is matrix math.

329. How do you compare two arrays element-wise?
     → Use operators like `==`, `>`, etc., on the arrays.

330. How do you apply mathematical functions like `sin`, `cos`, `exp` on arrays?
     → Use NumPy’s universal functions: `np.sin(arr)`, `np.cos(arr)`, `np.exp(arr)`.

#### **Universal Functions (ufuncs)**

331. What are universal functions in NumPy?
     → They’re fast, element-wise functions built in C for arrays.

332. Give examples of common ufuncs.
     → `sin`, `cos`, `exp`, `sqrt`, `add`, `subtract`.

333. How do ufuncs improve performance over Python loops?
     → They run vectorized code under the hood instead of slow Python loops.

334. What is the difference between `np.add` and `+` operator?
     → They behave the same; `np.add` is just the ufunc version.

335. How do you apply multiple ufuncs in a chain?
     → Just stack them: `np.sqrt(np.exp(arr))`.

336. How do you use `np.vectorize()` for custom functions?
     → Wrap your function: `v = np.vectorize(func)` and call it on arrays.

337. How do you compute element-wise comparisons using ufuncs?
     → Use comparison ufuncs like `np.equal`, `np.less`, etc.

338. How do you handle NaN values in ufunc operations?
     → Use `np.nan*` versions like `np.nansum` or clean them beforehand.

339. How do you reduce, accumulate, or outer-product using ufuncs?
     → Use `ufunc.reduce()`, `ufunc.accumulate()`, and `ufunc.outer()`.

340. Explain broadcasting rules with ufuncs.
     → Smaller shapes stretch to match bigger ones if dimensions are 1 or missing.

#### **Multidimensional Arrays**

341. How do you create a 3D array?
     → Use nested lists inside `np.array()` or use functions like `np.zeros((x,y,z))`.

342. How do you access elements in a 3D array?
     → Use three indices like `arr[i, j, k]`.

343. How do you reshape, flatten, and transpose multidimensional arrays?
     → Use `reshape()`, `ravel()/flatten()`, and `transpose()`.

344. How do you swap axes of a multidimensional array?
     → Use `np.swapaxes(arr, a, b)`.

345. How do you concatenate arrays along different axes?
     → Use `np.concatenate([...], axis=n)`.

346. How do you split arrays into multiple sub-arrays?
     → Use `np.split(arr, parts, axis=n)`.

347. How do you stack arrays vertically and horizontally?
     → Use `np.vstack()` and `np.hstack()`.

348. How do you tile an array?
     → Use `np.tile(arr, reps)`.

349. How do you perform element-wise operations on multidimensional arrays?
     → Just apply operators or ufuncs directly; NumPy handles the rest.

350. How do you iterate efficiently over multidimensional arrays?
     → Use `np.nditer()` for smooth, fast iteration.

#### **Data Types**

351. How do you check the data type of a NumPy array?
     → Look at `arr.dtype` like checking its ID card.

352. How do you change the data type of an array?
     → Use `arr.astype(new_type)` to politely convert it.

353. Explain the difference between `float32` and `float64`.
     → `float64` is more precise and heavier; `float32` is lighter but less detailed.

354. What are structured arrays in NumPy?
     → Arrays that hold multiple named fields, like tiny spreadsheets.

355. How do you create an array with complex numbers?
     → Include complex values or use `dtype=complex`.

356. How does NumPy handle type promotion during operations?
     → It upgrades types automatically to avoid losing information.

357. How do you store boolean values in a NumPy array?
     → Use `dtype=bool` or let NumPy infer it.

358. How do you store integer values with different precisions?
     → Use types like `int8`, `int16`, `int32`, `int64`.

359. How do you use the `dtype` parameter during array creation?
     → Pass `dtype=...` in functions like `np.array()` or `np.zeros()`.

360. What is the default data type of a NumPy array if not specified?
     → Usually `float64`, the fancy full-precision default.

#### **Memory Layout & Performance**

361. How is memory organized in a NumPy array?
     → In one continuous block with elements stored uniformly.

362. What is the difference between C-contiguous and F-contiguous arrays?
     → C stores rows first; F stores columns first.

363. How do you check if an array is contiguous?
     → Inspect `arr.flags`.

364. What is a view vs a copy in NumPy arrays?
     → A view shares data; a copy has its own data.

365. How do slicing operations affect memory usage?
     → They create views, so they avoid extra memory.

366. How do you create a deep copy of an array?
     → Use `arr.copy()`.

367. How do you use `np.ascontiguousarray()` and `np.asfortranarray()`?
     → They convert arrays into C- or Fortran-style layouts.

368. How do you check the memory size of an array?
     → Use `arr.nbytes`.

369. How does broadcasting improve performance?
     → It avoids duplicating data by reusing shapes cleverly.

370. How can vectorized operations speed up computations compared to loops?
     → They run fast compiled code instead of slow Python loops.

#### **Linear Algebra with NumPy**

371. How do you compute the transpose of a matrix?
     → Use `arr.T`.

372. How do you compute the determinant of a matrix?
     → Use `np.linalg.det(arr)`.

373. How do you compute the inverse of a matrix?
     → Use `np.linalg.inv(arr)`.

374. How do you compute eigenvalues and eigenvectors?
     → Use `np.linalg.eig(arr)`.

375. How do you perform Singular Value Decomposition (SVD)?
     → Use `np.linalg.svd(arr)`.

376. How do you compute the trace of a matrix?
     → Use `np.trace(arr)`.

377. How do you solve linear systems using `np.linalg.solve()`?
     → Call `np.linalg.solve(A, b)`.

378. How do you compute the rank of a matrix?
     → Use `np.linalg.matrix_rank(arr)`.

379. How do you perform matrix multiplication using `@` operator?
     → Just write `A @ B`.

380. How do you check if a matrix is positive definite?
     → Try `np.linalg.cholesky(arr)` and see if it succeeds.

#### **Random Numbers**

381. How do you generate uniform random numbers in NumPy?
     → Use `np.random.rand()` or `np.random.uniform()`.

382. How do you generate normal (Gaussian) distributed numbers?
     → Use `np.random.randn()` or `np.random.normal()`.

383. How do you set a random seed for reproducibility?
     → Call `np.random.seed(value)`.

384. How do you shuffle an array randomly?
     → Use `np.random.shuffle(arr)`.

385. How do you choose random samples from an array?
     → Use `np.random.choice(arr)`.

386. How do you generate integers randomly within a range?
     → Use `np.random.randint(low, high)`.

387. How do you simulate a random dice roll?
     → Use `np.random.randint(1, 7)`.

388. How do you sample without replacement?
     → Use `np.random.choice(arr, size, replace=False)`.

389. How do you sample with replacement?
     → Use `np.random.choice(arr, size, replace=True)`.

390. How do you generate random numbers for multidimensional arrays?
     → Call functions like `np.random.rand(x, y, z)`.

#### **Advanced Indexing & Masking**

391. How do you use boolean masks to filter arrays?
     → Create a condition and use it like `arr[arr > 5]`.

392. How do you assign values to elements using a mask?
     → Do `arr[mask] = value`.

393. How do you find indices where a condition is True?
     → Use `np.where(condition)`.

394. How do you use `np.where()` for conditional selection?
     → Use `np.where(cond, x, y)` to pick between two choices.

395. How do you perform element-wise selection using `np.choose()`?
     → Give an index array and a list of options: `np.choose(idx, options)`.

396. How do you use `np.take()` and `np.put()` for advanced indexing?
     → `take()` extracts by indices; `put()` writes values at given indices.

397. How do you index arrays with another array?
     → Pass the index array directly: `arr[index_array]`.

398. How do you combine multiple masks?
     → Use `&`, `|`, and `~` for AND, OR, NOT.

399. How do you broadcast masks to multidimensional arrays?
     → Shape them so NumPy can stretch them over axes automatically.

400. How do you avoid copies when masking for performance?
     → Use views or in-place updates instead of creating new masked arrays.

---

### **Batch 5: Advanced NumPy & Scientific Computing (Q401–Q500)**

#### **Advanced Indexing**

401. What is fancy indexing in NumPy?
     → Selecting items using arrays/lists of indices instead of simple slices—like giving NumPy a shopping list.

402. How do you select multiple non-contiguous elements using an array of indices?
     → Pass a list/array of positions, and NumPy plucks them out one by one.

403. How do you assign values using fancy indexing?
     → Use the fancy index on the left side of `=` and NumPy fills those exact spots.

404. What are the differences between basic slicing and advanced indexing?
     → Slicing makes views (no copying), while advanced indexing makes fresh copies.

405. How do you use boolean indexing for conditional assignment?
     → Create a `True/False` mask and assign values only where the mask is `True`.

406. How do you combine fancy and boolean indexing?
     → Chain them: first filter with a mask, then pick specific positions from the filtered set.

407. How do you select specific rows and columns simultaneously?
     → Use a tuple like `arr[row_indices, col_indices]` to grab the exact intersections.

408. What happens when you index with repeated values?
     → NumPy returns duplicates because it follows the index list faithfully.

409. How do you mask elements in multidimensional arrays efficiently?
     → Build a boolean mask of matching shape and apply it directly to the whole array.

410. How do you use `np.ix_()` for cross indexing?
     → Feed it row and column index lists, and it builds a grid so you get all combinations cleanly.

#### **Structured Arrays**

411. What is a structured array in NumPy?
     → An array that stores mixed-type fields in each element, like tiny rows of a table.

412. How do you define a structured array with multiple data types?
     → Provide a `dtype` with named fields and their individual types.

413. How do you access fields in a structured array?
     → Use the field name like `arr['age']` or `arr['score']`.

414. How do you perform computations on specific fields?
     → Extract the field and operate on it just like a normal NumPy array.

415. How do you sort structured arrays by a field?
     → Call `np.sort` or `.sort()` with the `order='fieldname'` argument.

416. How do you convert structured arrays to regular arrays?
     → View or cast them with `.view()` or `arr.tolist()` depending on the needed format.

417. How do you create a record array (`np.recarray`)?
     → Use `np.rec.array()` or convert an existing structured array with `.view(np.recarray)`.

418. How is a record array different from a structured array?
     → Record arrays let you access fields as attributes (e.g., `arr.age`) instead of only by keys.

419. How do you read structured arrays from files?
     → Use `np.genfromtxt` or `np.loadtxt` with a structured `dtype`.

420. How do you handle missing data in structured arrays?
     → Use masked arrays, sentinel values (like `-1`), or `np.nan` for floating-point fields.

#### **Linear Algebra**

421. How do you compute matrix multiplication using `np.matmul()`?
     → Call `np.matmul(A, B)` and it multiplies them following classic matrix rules.

422. What is the difference between `np.dot()` and `np.matmul()`?
     → `matmul` is strictly matrix-style; `dot` mixes behaviors depending on shapes.

423. How do you compute the determinant of a matrix?
     → Use `np.linalg.det(matrix)` and it hands you the determinant.

424. How do you compute the inverse of a matrix?
     → Call `np.linalg.inv(matrix)` if the matrix isn’t singular.

425. How do you compute eigenvalues and eigenvectors using `np.linalg.eig()`?
     → Feed the matrix to `np.linalg.eig()` and it returns values and vectors side by side.

426. How do you perform Singular Value Decomposition (SVD)?
     → Use `np.linalg.svd(matrix)` and it breaks the matrix into U, S, and Vᵀ.

427. How do you compute the pseudo-inverse of a matrix?
     → Call `np.linalg.pinv(matrix)` for a Moore–Penrose pseudo-inverse.

428. How do you solve linear equations with `np.linalg.solve()`?
     → Provide the coefficient matrix and the result vector: `np.linalg.solve(A, b)`.

429. How do you compute matrix norms?
     → Use `np.linalg.norm(matrix, ord=...)` with the desired norm type.

430. How do you check if a matrix is symmetric or positive definite?
     → Symmetry: check `A == A.T`; positive definite: verify all eigenvalues are positive.

#### **Random Sampling & Distributions**

431. How do you generate samples from a normal distribution?
     → Use `np.random.normal(mean, std, size)` to get bell-curve numbers.

432. How do you generate samples from a uniform distribution?
     → Call `np.random.uniform(low, high, size)` for evenly spread values.

433. How do you generate samples from a binomial distribution?
     → Use `np.random.binomial(n, p, size)` to simulate success counts.

434. How do you generate samples from a Poisson distribution?
     → Call `np.random.poisson(lam, size)` for event-count samples.

435. How do you generate samples from a custom probability distribution?
     → Use `np.random.choice(values, size, p=probabilities)`.

436. How do you set the seed for reproducibility?
     → Run `np.random.seed(number)` before generating randomness.

437. How do you shuffle an array randomly?
     → Call `np.random.shuffle(arr)` to mix it in place.

438. How do you select random samples without replacement?
     → Use `np.random.choice(arr, size, replace=False)`.

439. How do you select random samples with replacement?
     → Same `np.random.choice`, but set `replace=True`.

440. How do you generate multidimensional random arrays?
     → Pass a tuple as `size`, like `np.random.randn(3, 4)` for 3×4 randoms.

#### **Vectorized Operations**

441. What is vectorization in NumPy?
     → Doing operations on whole arrays at once instead of looping element by element.

442. Why is vectorization faster than loops?
     → Because NumPy runs optimized C-level operations under the hood.

443. How do you apply arithmetic operations to entire arrays?
     → Just use operators directly: `arr1 + arr2`, `arr * 5`, etc.

444. How do you use `np.where()` for conditional operations?
     → Provide a condition, a value-if-true, and value-if-false: `np.where(cond, a, b)`.

445. How do you perform element-wise maximum or minimum?
     → Use `np.maximum(a, b)` or `np.minimum(a, b)`.

446. How do you compute cumulative sum and cumulative product efficiently?
     → Call `np.cumsum(arr)` and `np.cumprod(arr)`.

447. How do you apply a custom function to an array using `np.vectorize()`?
     → Wrap your Python function with `np.vectorize(func)` and call it on the array.

448. How do you combine multiple vectorized operations efficiently?
     → Chain them together in one expression so NumPy optimizes the workflow.

449. How do you avoid temporary arrays during computations?
     → Use in-place ops like `arr += other` or specify `out=` in NumPy functions.

450. How do broadcasting rules affect vectorized operations?
     → They let arrays of different shapes interact by stretching smaller ones to match.

#### **Fourier Transform & Signal Processing Basics**

451. How do you compute the Fast Fourier Transform (FFT) in NumPy?
     → Use `np.fft.fft(signal)` to jump from time to frequency land.

452. How do you compute the inverse FFT?
     → Call `np.fft.ifft(freq_data)` to hop back to the time domain.

453. How do you extract the real and imaginary parts of an FFT result?
     → Use `.real` and `.imag` attributes on the complex array.

454. How do you compute the magnitude spectrum from FFT?
     → Take `np.abs(fft_result)` for the strength of each frequency.

455. How do you perform FFT on multidimensional arrays?
     → Use `np.fft.fftn(arr)` for N-D transforms.

456. How do you apply a window function before FFT?
     → Multiply the signal by a window (like Hanning) before calling FFT.

457. How do you filter signals using frequency-domain techniques?
     → FFT → zero/attenuate unwanted frequencies → inverse FFT.

458. How do you compute the convolution of two signals?
     → Use `np.convolve(a, b)` or FFT-based convolution.

459. How do you compute the correlation of two signals?
     → Use `np.correlate(a, b)` with the proper mode.

460. How do you perform zero-padding for FFT?
     → Pass a larger `n` to `np.fft.fft(signal, n)` to extend with zeros.

#### **Sparse Data Handling**

461. How do you represent sparse matrices in Python?
     → Use SciPy’s `scipy.sparse` module, which stores only the non-zero bits.

462. What is the difference between dense and sparse matrices?
     → Dense stores everything; sparse stores only the meaningful non-zero stuff.

463. How do you convert a dense NumPy array to a sparse matrix?
     → Wrap it with something like `csr_matrix(arr)`.

464. How do you convert a sparse matrix to a dense array?
     → Call `.toarray()` or `.todense()` on the sparse object.

465. What are the common sparse formats (CSR, CSC, COO)?
     → CSR = row-based, CSC = column-based, COO = simple coordinate list.

466. How do you perform arithmetic operations on sparse matrices?
     → Use normal operators (`+`, `-`, `*`) and SciPy handles the smart bookkeeping.

467. How do you perform matrix multiplication with sparse matrices?
     → Call `.dot()` or use `@` and let the sparse engine do the heavy lifting.

468. How do you save and load sparse matrices efficiently?
     → Use `scipy.sparse.save_npz()` and `load_npz()`.

469. When is using sparse matrices beneficial?
     → When most entries are zeros and memory shouldn’t be wasted.

470. How do you apply element-wise functions on sparse matrices?
     → Apply the function only to the `.data` array so you don’t blow up the sparsity.

#### **Memory & Performance Optimization**

471. How do you profile memory usage in NumPy arrays?
     → Check `arr.nbytes` or use tools like `sys.getsizeof()`.

472. How do you reduce memory usage with `float32` instead of `float64`?
     → Cast the array using `arr.astype('float32')`.

473. How do views and copies affect memory efficiency?
     → Views share memory; copies take fresh space.

474. How do you avoid unnecessary copies during computations?
     → Use in-place ops like `arr += x` or functions with `out=`.

475. How do you use `np.memmap` for large datasets?
     → Create a memory-mapped file so data loads in chunks instead of fully.

476. How do you handle out-of-core computation with NumPy?
     → Combine `memmap` with chunked processing or use tools like Dask.

477. How does broadcasting reduce memory overhead?
     → It fakes expansion without actually creating big duplicated arrays.

478. How do you pre-allocate arrays for performance?
     → Allocate once with `np.empty()` or `np.zeros()` and fill it later.

479. How do you use `np.einsum()` for optimized operations?
     → Provide Einstein notation to perform complex ops without extra temporaries.

480. How do you parallelize NumPy computations (basic approaches)?
     → Use libraries like Numba, multiprocessing, or vectorized BLAS under the hood.

#### **Interfacing NumPy with C/C++**

481. What is Cython, and how can it speed up NumPy operations?
     → A Python-to-C compiler that speeds things up by adding static types and tight loops.

482. How do you convert NumPy arrays to C arrays?
     → Access the underlying buffer via `.ctypes` or obtain pointers using `arr.ctypes.data`.

483. How do you use `ctypes` to pass NumPy arrays to C functions?
     → Match the C signature, ensure contiguous memory, and pass `arr.ctypes.data_as(...)`.

484. How do you use `cffi` with NumPy?
     → Create a buffer interface and hand the array’s memory pointer to CFFI’s foreign functions.

485. What is Numba, and how does it optimize NumPy code?
     → A JIT compiler that turns Python+NumPy loops into fast machine code.

486. How do you jit-compile a function using Numba?
     → Decorate it with `@njit` and call it normally.

487. How do you handle multidimensional arrays in C/C++ extensions?
     → Use the NumPy C-API to read shape, strides, and data pointers safely.

488. How do you avoid memory copying between NumPy and C/C++?
     → Work directly with the NumPy buffer using contiguous arrays and pointer access.

489. How do you benchmark Cython/Numba code against pure Python?
     → Use `timeit` or `perf_counter` and compare runtimes across implementations.

490. Give an example of a computationally heavy operation that benefits from C/C++ interfacing.
     → Large nested loops like finite-difference simulations or particle-based physics updates.

#### **Advanced NumPy Functions**

491. How do you compute the Kronecker product of two arrays?
     → Use `np.kron(A, B)` to tile one array across the other.

492. How do you perform matrix decomposition using NumPy?
     → Call functions like `np.linalg.svd`, `np.linalg.qr`, or `np.linalg.cholesky`.

493. How do you compute covariance and correlation matrices?
     → Use `np.cov(data)` and `np.corrcoef(data)`.

494. How do you compute eigenvalues of a covariance matrix?
     → Feed it to `np.linalg.eig()`.

495. How do you perform principal component analysis (PCA) using NumPy?
     → Center data → compute covariance → eig-decompose → project onto top eigenvectors.

496. How do you normalize arrays efficiently?
     → Subtract mean and divide by standard deviation in vectorized form.

497. How do you compute moving averages using NumPy?
     → Use convolution: `np.convolve(arr, np.ones(n)/n, mode='valid')`.

498. How do you handle missing values in NumPy arrays?
     → Use masks, `np.nan*` functions, or replace with sentinel values.

499. How do you perform interpolation with NumPy arrays?
     → Use `np.interp(x, xp, fp)` for simple 1-D interpolation.

500. How do you integrate NumPy with other scientific Python libraries efficiently?
     → Share arrays directly since most libraries accept NumPy buffers natively.

---

### **Batch 6: Pandas Basics & Data Manipulation (Q501–Q600)**

#### **Introduction to Pandas**

501. What is Pandas, and why is it used in Python?
     → A data-handling library that makes working with tables super easy.

502. How do you import Pandas?
     → `import pandas as pd`.

503. What are the main data structures in Pandas?
     → Series (1-D) and DataFrame (2-D).

504. What is the difference between a Series and a DataFrame?
     → Series is a single labeled column; DataFrame is a whole table of them.

505. How do you create a Series from a list, dictionary, or array?
     → Pass the object to `pd.Series(...)`.

506. How do you create a DataFrame from a dictionary of lists?
     → Use `pd.DataFrame(your_dict)`.

507. How do you create a DataFrame from a NumPy array?
     → Feed the array into `pd.DataFrame(arr)` with optional column names.

508. How do you view the first few rows of a DataFrame?
     → Call `df.head()`.

509. How do you view the last few rows of a DataFrame?
     → Call `df.tail()`.

510. How do you get basic information about a DataFrame using `.info()`?
     → Run `df.info()` to see columns, types, and memory use.

#### **Indexing & Selection**

511. How do you access a single column in a DataFrame?
     → Use `df['col']` or `df.col`.

512. How do you access multiple columns?
     → Pass a list like `df[['col1', 'col2']]`.

513. How do you access a single row using `iloc`?
     → Use `df.iloc[row_number]`.

514. How do you access a single row using `loc`?
     → Use `df.loc[row_label]`.

515. How do you slice rows using `iloc`?
     → Use numeric slices like `df.iloc[2:7]`.

516. How do you slice rows using `loc`?
     → Slice by labels like `df.loc['a':'d']`.

517. How do you select specific rows and columns together?
     → Use `df.loc[row_sel, col_sel]` or `df.iloc[row_sel, col_sel]`.

518. How do you filter rows based on column values?
     → Write a condition like `df[df['age'] > 30]`.

519. How do you use boolean indexing for filtering?
     → Build a True/False mask and apply it directly: `df[mask]`.

520. How do you reset and set index in a DataFrame?
     → Use `df.reset_index()` and `df.set_index('col')`.

#### **Data Cleaning**

521. How do you check for missing values in a DataFrame?
     → Use `df.isna()` or `df.isnull()` to spot the sneaky blanks.

522. How do you drop rows with missing values?
     → `df.dropna()` tosses out any row with gaps.

523. How do you drop columns with missing values?
     → `df.dropna(axis=1)` wipes out whole columns full of emptiness.

524. How do you fill missing values with a constant or computed value?
     → `df.fillna(value)` pops in whatever you choose.

525. How do you handle duplicate rows in a DataFrame?
     → Use `df.duplicated()` to find them and `df.drop_duplicates()` to boot them out.

526. How do you convert a column to a different data type?
     → `df['col'] = df['col'].astype(new_type)` gives it a fresh identity.

527. How do you rename columns and indices?
     → `df.rename(columns=..., index=...)` swaps in new names neatly.

528. How do you remove leading/trailing whitespace from column names?
     → `df.columns = df.columns.str.strip()` gives them a tidy haircut.

529. How do you detect and remove outliers?
     → Compute limits (like z-scores or IQR) and filter anything that strays too far.

530. How do you handle categorical data in Pandas?
     → Convert with `astype('category')` to save memory and speed things up.

#### **Merging, Joining & Concatenation**

531. How do you concatenate two DataFrames vertically?
     → Use `pd.concat([df1, df2], axis=0)` to stack them top to bottom.

532. How do you concatenate two DataFrames horizontally?
     → Use `pd.concat([df1, df2], axis=1)` to place them side by side.

533. What is the difference between `concat()` and `append()`?
     → `append()` is just a simpler, slower wrapper; `concat()` is the real flexible tool.

534. How do you merge two DataFrames on a common column?
     → Use `pd.merge(df1, df2, on='col')`.

535. How do you perform an inner join using `merge()`?
     → Add `how='inner'` in `pd.merge(...)`.

536. How do you perform a left, right, and outer join?
     → Set `how='left'`, `how='right'`, or `how='outer'` in `merge()`.

537. How do you join DataFrames using indices instead of columns?
     → Use `df1.join(df2)` or `merge(left_index=True, right_index=True)`.

538. How do you handle overlapping column names while merging?
     → Use `suffixes=('_x', '_y')` to keep them distinguishable.

539. How do you merge multiple DataFrames at once?
     → Chain `merge()` calls or fold them with a loop.

540. How do you merge on multiple keys?
     → Provide a list: `pd.merge(df1, df2, on=['key1', 'key2'])`.

#### **Grouping & Aggregation**

541. What is the purpose of `groupby()` in Pandas?
     → It clusters rows into little buckets so you can summarize each group.

542. How do you group data by a single column?
     → Call `df.groupby('col')`.

543. How do you group data by multiple columns?
     → Pass a list: `df.groupby(['col1', 'col2'])`.

544. How do you compute summary statistics after grouping?
     → Chain `.mean()`, `.sum()`, `.count()`, or similar on the groupby object.

545. How do you apply multiple aggregation functions to a group?
     → Use `.agg(['mean', 'max', 'min'])` or any mix you like.

546. How do you apply a custom aggregation function?
     → Pass your own function inside `.agg(func)`.

547. How do you filter groups based on a condition?
     → Use `.filter(lambda g: condition)` to keep only the worthy groups.

548. How do you iterate over groups?
     → Loop with `for name, group in df.groupby(col): ...`.

549. How do you transform groups without reducing the size?
     → Use `.transform(func)`—it returns something the same length as the original.

550. How do you unstack or pivot grouped data?
     → Call `.unstack()` to turn grouped levels into columns.

#### **Pivot Tables**

551. What is a pivot table in Pandas?
     → A table that rearranges data by summarizing values across chosen rows and columns.

552. How do you create a pivot table using `pivot_table()`?
     → Use `pd.pivot_table(df, ...)` with your chosen settings.

553. How do you specify index, columns, and values in a pivot table?
     → Pass them as `index=`, `columns=`, and `values=` in `pivot_table()`.

554. How do you aggregate data in a pivot table using different functions?
     → Use `aggfunc=` with a function or list of functions.

555. How do you handle missing values in pivot tables?
     → Fill them using `fill_value=` inside `pivot_table()`.

556. How do you normalize data in a pivot table?
     → Divide by row or column totals after creation for scaled results.

557. How do you group data and create pivot tables simultaneously?
     → Group first, then pivot the grouped summary.

558. How do you reshape a DataFrame using `melt()`?
     → `pd.melt(df)` turns wide tables into long, tidy form.

559. How do you convert long format to wide format using `pivot()`?
     → Use `df.pivot(index, columns, values)` to spread values across columns.

560. How do you combine pivot tables with `groupby()` results?
     → Group, aggregate, then feed the aggregated result into a pivot or vice versa.

#### **Reading & Writing Data**

561. How do you read a CSV file into a DataFrame?
     → Use `pd.read_csv('file.csv')`.

562. How do you write a DataFrame to a CSV file?
     → `df.to_csv('file.csv', index=False)`.

563. How do you read an Excel file into a DataFrame?
     → Use `pd.read_excel('file.xlsx')`.

564. How do you write a DataFrame to an Excel file?
     → `df.to_excel('file.xlsx', index=False)`.

565. How do you read a JSON file into a DataFrame?
     → `pd.read_json('file.json')`.

566. How do you write a DataFrame to a JSON file?
     → `df.to_json('file.json')`.

567. How do you read from SQL using Pandas?
     → Use `pd.read_sql(query, connection)`.

568. How do you write to SQL from a DataFrame?
     → `df.to_sql('table', connection, if_exists='append')`.

569. How do you handle different delimiters in CSV files?
     → Pass `sep=';'` or another delimiter to `read_csv()`.

570. How do you read only specific columns or rows from a file?
     → Use `usecols=` for columns and `nrows=` or `skiprows=` for rows.

#### **Indexing & Selection (Advanced)**

571. How do you select rows based on multiple conditions?
     → Combine conditions with `&` and `|` inside `df[...]`.

572. How do you use the `query()` method for filtering?
     → Write conditions as strings: `df.query("age > 30 and city == 'NY'")`.

573. How do you select rows using `isin()`?
     → `df[df['col'].isin([values])]`.

574. How do you select rows using `between()`?
     → `df[df['col'].between(a, b)]`.

575. How do you use `str` methods on string columns?
     → Call `df['col'].str.method()` like `str.contains()` or `str.upper()`.

576. How do you use `dt` methods on datetime columns?
     → Use `df['date'].dt.year`, `.dt.month`, `.dt.day`, etc.

577. How do you sort a DataFrame by one or multiple columns?
     → Use `df.sort_values(['col1', 'col2'])`.

578. How do you sort a DataFrame by index?
     → Call `df.sort_index()`.

579. How do you rank data in a column?
     → Use `df['col'].rank()`.

580. How do you sample rows randomly from a DataFrame?
     → `df.sample(n=...)` or `df.sample(frac=...)`.

#### **Data Transformation**

581. How do you add a new column based on existing columns?
     → Create it directly: `df['new'] = df['a'] + df['b']`.

582. How do you delete a column from a DataFrame?
     → Use `df.drop('col', axis=1)`.

583. How do you rename specific columns?
     → Use `df.rename(columns={'old': 'new'})`.

584. How do you apply a function to a column using `apply()`?
     → `df['col'].apply(func)`.

585. How do you apply a function to each element using `map()`?
     → `df['col'].map(func_or_dict)`.

586. How do you apply a function to multiple columns using `apply()`?
     → `df[['a','b']].apply(func, axis=1)`.

587. How do you replace values in a column?
     → `df['col'].replace(old, new)`.

588. How do you convert wide-format data to long format using `melt()`?
     → Use `pd.melt(df, id_vars=..., value_vars=...)`.

589. How do you normalize a column of numeric data?
     → `(df['col'] - df['col'].mean()) / df['col'].std()`.

590. How do you create dummy/indicator variables from categorical columns?
     → Use `pd.get_dummies(df['col'])`.

#### **Time Series Basics**

591. How do you convert a column to datetime format?
     → Use `pd.to_datetime(df['col'])`.

592. How do you set a datetime column as index?
     → `df.set_index('date_col', inplace=True)`.

593. How do you filter rows based on date ranges?
     → Slice with `df.loc['2021-01':'2021-06']` or compare with conditions.

594. How do you extract year, month, or day from datetime columns?
     → Use `.dt.year`, `.dt.month`, `.dt.day`.

595. How do you resample time series data to a different frequency?
     → `df.resample('M').mean()` or another rule/function combo.

596. How do you compute rolling averages?
     → `df['col'].rolling(window).mean()`.

597. How do you compute cumulative sums over time?
     → `df['col'].cumsum()`.

598. How do you shift and lag time series data?
     → Use `df['col'].shift(n)`.

599. How do you handle missing dates in a time series?
     → Reindex with a full date range and fill missing values.

600. How do you merge or join time series datasets effectively?
     → Align on datetime indices using `merge`, `join`, or aligned reindexing.

---

### **Batch 7: Advanced Pandas & Performance (Q601–Q700)**

#### **MultiIndex & Hierarchical Indexing**

601. What is a MultiIndex in Pandas?
     → A fancy “multi-level label system” that lets your rows or columns have more than one level of indexing.

602. How do you create a MultiIndex DataFrame?
     → Use lists/arrays of tuples or `pd.MultiIndex.from_product()`/`from_tuples()` to build the layered index.

603. How do you set multiple columns as index?
     → Call `df.set_index(["col1", "col2"])`.

604. How do you access data using a MultiIndex?
     → Use `.loc[]` with tuples that match the index levels.

605. How do you swap levels in a MultiIndex?
     → Use `df.swaplevel()`.

606. How do you sort a MultiIndex DataFrame?
     → Use `df.sort_index()`.

607. How do you reset a MultiIndex to columns?
     → Use `df.reset_index()`.

608. How do you slice data in a MultiIndex DataFrame?
     → Use `pd.IndexSlice` with `.loc[]` for clean multi-level slicing.

609. How do you select rows at a specific level of a MultiIndex?
     → Use `.xs(value, level="level_name")`.

610. How do you aggregate data over a specific level in MultiIndex?
     → Use `.groupby(level="level_name").agg(...)`.

#### **Reshaping & Pivoting**

611. How do you reshape a DataFrame from wide to long format?
     → Use `melt()` to squeeze many columns into two tidy ones.

612. How do you reshape a DataFrame from long to wide format?
     → Use `pivot()` or `pivot_table()` to spread rows back into columns.

613. How do you stack and unstack a DataFrame?
     → `stack()` pushes columns into rows, `unstack()` pulls row levels into columns.

614. How do you pivot a DataFrame with `pivot()`?
     → Call `df.pivot(index=..., columns=..., values=...)` like assembling a tiny spreadsheet.

615. How do you create pivot tables with `pivot_table()`?
     → Use `pivot_table()` with an aggregation function for smarter summarising.

616. How do you handle missing values during pivoting?
     → Use `fill_value=` or clean with `dropna()` before or after the pivot.

617. How do you merge reshaping and aggregation operations?
     → Combine `groupby()` with `pivot_table()` or `agg()` in one tidy chain.

618. How do you melt multiple columns simultaneously?
     → Pass them in `value_vars` of `melt()` to melt a whole bunch at once.

619. How do you reorder levels after unstacking?
     → Call `reorder_levels()` or `swaplevel()` like rearranging index furniture.

620. How do you combine stacked and unstacked operations efficiently?
     → Chain `stack()` and `unstack()` with `sort_index()` for neat, speedy reshaping.

#### **Time Series Analysis**

621. How do you generate a date range in Pandas?
     → Use `pd.date_range()` to pop out a neat sequence of dates like a date-making machine.

622. How do you resample time series data?
     → Use `.resample()` to squish or stretch your timeline into new time buckets.

623. How do you perform rolling and expanding window calculations?
     → Use `.rolling()` for sliding windows and `.expanding()` for ever-growing ones.

624. How do you shift or lag a time series?
     → Use `.shift()` to nudge data forward or backward in time.

625. How do you handle time zones in a time series?
     → Use `.tz_localize()` and `.tz_convert()` to assign or change time zones.

626. How do you interpolate missing values in time series?
     → Use `.interpolate()` to gently fill the gaps between timestamps.

627. How do you compute cumulative statistics over time?
     → Use `.cumsum()`, `.cumprod()`, or `.cummax()` like building stats layer by layer.

628. How do you compute exponentially weighted moving averages?
     → Use `.ewm().mean()` for a smooth average that “forgets” old stuff slowly.

629. How do you handle irregularly spaced time series?
     → Reindex or resample to regular intervals so your timeline stops being chaotic.

630. How do you merge or join time series datasets effectively?
     → Use `merge_asof()` to align them by nearest timestamps like a clever time-matching puzzle.

#### **Memory Optimization**

631. How do you check the memory usage of a DataFrame?
     → Use `df.memory_usage(deep=True)` to see where the bytes are hiding.

632. How do you convert object columns to categorical to save memory?
     → Use `df["col"] = df["col"].astype("category")`.

633. How do you downcast numeric columns for memory efficiency?
     → Use `pd.to_numeric(..., downcast="integer/float")`.

634. How do you use sparse data structures in Pandas?
     → Convert with `astype("Sparse[...]")` so zeros stop hogging space.

635. How do you drop unused columns to optimize memory?
     → Use `df.drop(columns=[...])` to toss out the dead weight.

636. How do you optimize datetime columns for memory usage?
     → Parse once with `to_datetime()` and avoid storing messy string versions.

637. How do you handle large CSV files efficiently?
     → Use `chunksize`, `dtype` hints, or compression to keep things manageable.

638. How do you read a subset of columns to save memory?
     → Use `usecols=` when reading so you skip the extra baggage.

639. How do you use chunking while reading large datasets?
     → Load in pieces with `pd.read_csv(..., chunksize=...)` like eating a big meal in small bites.

640. How do you combine memory optimization with performance optimization?
     → Pick efficient dtypes, drop junk early, and process in chunks for a smooth and speedy workflow.

#### **Vectorized Operations**

641. How do you perform element-wise operations on DataFrames?
     → Use vectorized operators like `+`, `-`, `*`, or functions that act on whole arrays at once.

642. How do you use `apply()` for row-wise or column-wise computations?
     → Use `df.apply(func, axis=0/1)` to run your function across rows or columns.

643. How do you use `map()` for Series transformations?
     → Use `series.map(func_or_dict)` to transform each value one by one.

644. How do you use `applymap()` for element-wise operations on DataFrames?
     → Use `df.applymap(func)` to touch every single cell with your function.

645. How do you replace loops with vectorized operations?
     → Use NumPy-style operations that work on whole columns at once instead of slow Python loops.

646. How do you perform conditional operations efficiently using `np.where()`?
     → Use `np.where(condition, value1, value2)` for fast if-else logic.

647. How do you compute cumulative statistics efficiently?
     → Use built-ins like `.cumsum()`, `.cummax()`, `.cummin()`, or `.cumprod()`.

648. How do you compute group-wise transformations using `transform()`?
     → Use `groupby().transform()` to reshape values while keeping the original DataFrame shape.

649. How do you combine vectorized operations with aggregation?
     → Apply fast column operations first, then summarise with `groupby().agg()`.

650. How do you handle missing values efficiently in vectorized operations?
     → Use helpers like `.fillna()`, `.isna()`, or NumPy functions that naturally skip or handle NaNs.

#### **Advanced Joins & Merging**

651. How do you perform many-to-many joins in Pandas?
     → Use `merge()` on non-unique keys and Pandas will expand all matching combinations.

652. How do you perform fuzzy matching while joining?
     → Use libraries like `fuzzywuzzy`/`thefuzz` with `apply()` to match closest strings before merging.

653. How do you merge DataFrames on multiple keys?
     → Pass a list to `on=[...]` or `left_on`/`right_on`.

654. How do you perform cross joins?
     → Use `how="cross"` in `merge()` for a full Cartesian product.

655. How do you merge using indices instead of columns?
     → Set `left_index=True` and/or `right_index=True`.

656. How do you handle overlapping column names during merge?
     → Use `suffixes=('_x', '_y')` or custom ones to keep things tidy.

657. How do you merge with different join types (inner, outer, left, right)?
     → Set `how=` to `"inner"`, `"outer"`, `"left"`, or `"right"`.

658. How do you concatenate multiple DataFrames efficiently?
     → Use `pd.concat([...])` and set `ignore_index=True` if you want a clean index.

659. How do you join on datetime indices?
     → Make sure both are datetime-indexed and then use `.join()` or `merge()` with index flags.

660. How do you validate merge operations for consistency?
     → Use the `validate=` parameter like `"one_to_one"` or `"one_to_many"` to catch mistakes early.

#### **Advanced Aggregation & GroupBy**

661. How do you apply multiple aggregation functions to a group?
     → Use `groupby().agg({'col': ['sum','mean',...]})` to stack several stats at once.

662. How do you apply custom aggregation functions?
     → Pass your own function into `agg()` like `groupby().agg(my_func)`.

663. How do you filter groups using `filter()`?
     → Use `groupby().filter(lambda g: condition)` to keep only the groups you want.

664. How do you transform groups without reducing size?
     → Use `groupby().transform()` so the output matches the original shape.

665. How do you rank items within groups?
     → Use `groupby().rank()` to rank values independently inside each group.

666. How do you compute rolling statistics within groups?
     → Use `groupby().rolling(window).mean()` then reset the index if needed.

667. How do you handle hierarchical indexes while aggregating?
     → Aggregate first, then tidy with `reset_index()` or `reorder_levels()`.

668. How do you pivot grouped data for analysis?
     → Group, aggregate, then run `pivot()` or `pivot_table()` to reshape it.

669. How do you compute weighted averages within groups?
     → Use `groupby().apply(lambda g: np.average(g[col], weights=g[w]))`.

670. How do you visualize group-wise statistics efficiently?
     → Pre-aggregate with `groupby().agg()` and plot the clean result directly.

#### **Categorical Data**

671. How do you convert a column to categorical type?
     → Use `df["col"] = df["col"].astype("category")`.

672. How do you reorder categories in a categorical column?
     → Use `.cat.reorder_categories([...], ordered=True)`.

673. How do you add or remove categories?
     → Use `.cat.add_categories()` or `.cat.remove_categories()`.

674. How do you compute group statistics using categorical columns?
     → Treat them like normal group keys with `groupby()`.

675. How do you use categorical columns for performance optimization?
     → Convert repetitive text to categories so lookups and comparisons get faster.

676. How do you handle missing values in categorical columns?
     → Use `.fillna()` or even create a “missing” category if it helps.

677. How do you merge datasets with categorical columns efficiently?
     → Ensure both sides share the same categories so matching is quick.

678. How do you perform one-hot encoding with `get_dummies()`?
     → Use `pd.get_dummies(df["col"])` to explode categories into neat binary columns.

679. How do you handle ordinal categorical variables?
     → Create categories with `ordered=True` and a proper order list.

680. How do categorical data types affect memory usage?
     → They shrink repeated text into tiny numeric codes, saving loads of space.

#### **String Operations**

681. How do you use vectorized string methods with `.str` accessor?
     → Call things like `df["col"].str.lower()` to process all strings at once.

682. How do you extract substrings using `.str.extract()`?
     → Use regex patterns inside `.str.extract(r"...")` to pull out matching pieces.

683. How do you replace patterns using `.str.replace()`?
     → Use `.str.replace(old, new, regex=True)` for quick text swaps.

684. How do you check for string presence with `.str.contains()`?
     → Use `.str.contains("text")` to spot rows that include a pattern.

685. How do you handle missing values in string columns?
     → Use `.fillna("")` or skip them with `na=False` in string methods.

686. How do you split strings into multiple columns?
     → Use `.str.split(..., expand=True)` to break text into new columns.

687. How do you join multiple string columns efficiently?
     → Use `.str.cat([...], sep=" ")` to glue them together.

688. How do you apply custom string functions to a column?
     → Use `.apply(func)` when built-in `.str` tools aren’t enough.

689. How do you remove unwanted characters from strings?
     → Use `.str.replace(r"[chars]", "", regex=True)` to clean things up.

690. How do you handle Unicode and encoding issues in strings?
     → Decode early when reading data and use `.str.normalize()` to tidy Unicode.

#### **Performance Optimization Techniques**

691. How do you profile Pandas code for performance bottlenecks?
     → Use `%timeit`, `%prun`, or `cProfile` to spot slow sections.

692. How do you optimize DataFrame operations with vectorization?
     → Replace Python loops with NumPy-style whole-column operations.

693. How do you use `eval()` and `query()` for faster computation?
     → Use them to run expressions directly on columns with less overhead.

694. How do you use `numba` to accelerate Pandas operations?
     → Wrap custom numeric functions with `@numba.njit` for big speedups.

695. How do you handle large datasets efficiently in memory?
     → Load only what you need, downcast types, and process in chunks.

696. How do you use categorical and sparse data types for performance?
     → Convert repetitive or mostly-zero data to smaller, cheaper formats.

697. How do you avoid chained indexing warnings and improve speed?
     → Use `.loc[]` cleanly so Pandas knows exactly what you mean.

698. How do you parallelize operations with Dask or multiprocessing?
     → Break work into pieces and let multiple cores chew on them at once.

699. How do you reduce computation time using chunked processing?
     → Read and process data in bite-sized chunks instead of all at once.

700. How do you combine performance optimization with maintainable code?
     → Use clear pipelines, avoid clever-but-confusing tricks, and optimize only where it matters.

---

### **Batch 8: Matplotlib & Data Visualization (Q701–Q800)**

#### **Plotting Basics**

701. What is Matplotlib, and why is it used in Python?
     → It’s a plotting library used to make graphs and visualisations of data.

702. How do you import Matplotlib for plotting?
     → Use `import matplotlib.pyplot as plt`.

703. What is the difference between `pyplot` and the object-oriented interface?
     → `pyplot` is quick and simple, while the OO style gives more control and structure.

704. How do you create a simple line plot using `plt.plot()`?
     → Call `plt.plot(x, y)` and then `plt.show()`.

705. How do you create a scatter plot using `plt.scatter()`?
     → Use `plt.scatter(x, y)`.

706. How do you create a bar chart using `plt.bar()`?
     → Use `plt.bar(categories, values)`.

707. How do you create a histogram using `plt.hist()`?
     → Use `plt.hist(data)`.

708. How do you create a pie chart using `plt.pie()`?
     → Use `plt.pie(values)`.

709. How do you add labels to the x-axis and y-axis?
     → Use `plt.xlabel("label")` and `plt.ylabel("label")`.

710. How do you add a title to a plot?
     → Use `plt.title("your title")`.

#### **Customization of Plots**

711. How do you change line styles and colors?
     → Pass parameters like `linestyle="--"` and `color="red"` inside `plt.plot()`.

712. How do you change marker styles and sizes?
     → Use `marker="o"` and `markersize=10` in your plotting call.

713. How do you customize bar colors and widths?
     → Add `color="…" ` and `width=…` inside `plt.bar()`.

714. How do you add grid lines to a plot?
     → Call `plt.grid(True)`.

715. How do you add legends to plots?
     → Include `label="…" ` in plot calls and then use `plt.legend()`.

716. How do you set axis limits?
     → Use `plt.xlim(min, max)` and `plt.ylim(min, max)`.

717. How do you add annotations to specific points?
     → Use `plt.annotate("text", xy=(x, y))`.

718. How do you change font size and style for labels?
     → Pass parameters like `fontsize=…` and `fontstyle="…" ` to label functions.

719. How do you add multiple lines to a single plot?
     → Call `plt.plot()` repeatedly before `plt.show()`.

720. How do you save a plot to a file in PNG, PDF, or SVG format?
     → Use `plt.savefig("filename.png")` (or `.pdf`, `.svg`).

#### **Subplots and Figure Management**

721. How do you create multiple subplots using `plt.subplot()`?
     → Call `plt.subplot(rows, cols, index)` before each plot.

722. How do you use `plt.subplots()` for a grid of plots?
     → Use `fig, axes = plt.subplots(r, c)` to get a figure and array of axes.

723. How do you share axes between subplots?
     → Pass `sharex=True` or `sharey=True` to `plt.subplots()`.

724. How do you adjust spacing between subplots?
     → Use `plt.tight_layout()` or `fig.subplots_adjust()`.

725. How do you set figure size and resolution?
     → Use `plt.figure(figsize=(w, h), dpi=resolution)`.

726. How do you add titles to individual subplots?
     → Use `ax.set_title("title")` on each subplot axis.

727. How do you add a global title for the figure?
     → Use `fig.suptitle("main title")`.

728. How do you manage multiple figures simultaneously?
     → Create and switch using `plt.figure(id)`.

729. How do you save multiple subplots to a single file?
     → Save the whole figure with `plt.savefig("file.png")`.

730. How do you clear a figure using `plt.clf()` or `plt.close()`?
     → `plt.clf()` clears the figure; `plt.close()` closes it completely.

#### **Advanced Plots**

731. How do you create a stacked bar chart?
     → Plot bars with the same x-values and use the `bottom=` parameter to stack them.

732. How do you create a horizontal bar chart?
     → Use `plt.barh()` instead of `plt.bar()`.

733. How do you create a box plot using `plt.boxplot()`?
     → Call `plt.boxplot(data)`.

734. How do you create a violin plot?
     → Use `plt.violinplot(data)`.

735. How do you create an area plot?
     → Use `plt.stackplot(x, y_values)`.

736. How do you create an error bar plot?
     → Use `plt.errorbar(x, y, yerr=errors)`.

737. How do you create a scatter plot with varying sizes and colors?
     → Use `plt.scatter(x, y, s=sizes, c=colors)`.

738. How do you create a heatmap using Matplotlib?
     → Use `plt.imshow(data, aspect="auto")`.

739. How do you create a contour plot using `plt.contour()`?
     → Call `plt.contour(X, Y, Z)`.

740. How do you create 3D plots using `mpl_toolkits.mplot3d`?
     → Create an Axes3D object and use functions like `ax.plot3D()` or `ax.scatter3D()`.

#### **Figure and Axes Manipulation**

741. How do you access the figure and axes objects in a plot?
     → Use `fig, ax = plt.subplots()` to get them directly.

742. How do you modify axes limits and scales?
     → Use `ax.set_xlim()`, `ax.set_ylim()`, or `ax.set_xscale()` / `ax.set_yscale()`.

743. How do you add secondary axes to a plot?
     → Use `ax.secondary_xaxis()` or `ax.secondary_yaxis()`.

744. How do you set logarithmic scale for an axis?
     → Call `plt.xscale("log")` or `plt.yscale("log")`.

745. How do you customize tick labels and positions?
     → Use `ax.set_xticks()` and `ax.set_xticklabels()` (same for y).

746. How do you rotate tick labels for readability?
     → Use `plt.xticks(rotation=angle)`.

747. How do you hide or remove axes and spines?
     → Use `ax.spines[...] .set_visible(False)` or `plt.axis("off")`.

748. How do you add text and annotations using axes coordinates?
     → Use `ax.text(x, y, "text", transform=ax.transAxes)`.

749. How do you draw lines, rectangles, and circles manually?
     → Use patches like `Line2D`, `Rectangle`, or `Circle` and add with `ax.add_patch()`.

750. How do you layer multiple plots on the same axes?
     → Just call multiple plotting functions on the same `ax` object.

#### **Plot Styling & Themes**

751. How do you use Matplotlib styles (`plt.style.use()`)?
     → Call `plt.style.use("stylename")` to switch the whole plot’s look.

752. How do you customize the color cycle for lines?
     → Set `ax.set_prop_cycle(color=[...])` with your chosen list.

753. How do you use Seaborn styles with Matplotlib?
     → Import Seaborn and call `sns.set_theme()` or `sns.set_style()`.

754. How do you create transparent plots?
     → Use `plt.plot(..., alpha=value)` or save with `plt.savefig(..., transparent=True)`.

755. How do you control alpha transparency in plots?
     → Add `alpha=0.0–1.0` to any plotting call.

756. How do you apply custom colormaps?
     → Pass `cmap="colormap_name"` to functions that support it.

757. How do you create visually appealing color palettes?
     → Use palettes from Seaborn or `plt.cm` colormap families.

758. How do you control linewidth and marker edge color?
     → Use `linewidth=...` and `markeredgecolor="..."` in your plot call.

759. How do you use LaTeX formatting for labels?
     → Wrap text in `$...$` and ensure `plt.rcParams["text.usetex"]=True`.

760. How do you globally set figure parameters using `rcParams`?
     → Modify values in `plt.rcParams[...]` before plotting.

#### **Interactive Visualizations**

761. How do you create interactive plots in Jupyter notebooks?
     → Use `%matplotlib notebook` or `%matplotlib widget`.

762. How do you zoom, pan, and update plots interactively?
     → Use the toolbar that appears with interactive backends.

763. How do you use `plt.ion()` and `plt.ioff()` for interactive mode?
     → `plt.ion()` turns interactive updates on, `plt.ioff()` turns them off.

764. How do you update plots dynamically with live data?
     → Modify the data on an existing artist and call `plt.draw()` or `fig.canvas.draw()`.

765. How do you capture click events on a plot?
     → Connect to `fig.canvas.mpl_connect("button_press_event", handler)`.

766. How do you capture hover events on a plot?
     → Use `mpl_connect("motion_notify_event", handler)`.

767. How do you use sliders to adjust plot parameters interactively?
     → Use `matplotlib.widgets.Slider` and update the plot inside its callback.

768. How do you use buttons and widgets with Matplotlib?
     → Use widgets like `Button`, `CheckButtons`, and bind callback functions.

769. How do you embed Matplotlib plots in GUIs?
     → Use backends for Tkinter, PyQt, or wxPython with their canvas widgets.

770. How do you create animations using `FuncAnimation`?
     → Use `FuncAnimation(fig, update_function, frames=...)` to redraw frames.

#### **3D Plots & Advanced 3D Techniques**

771. How do you create a 3D scatter plot?
     → Use `ax.scatter3D(x, y, z)` on a 3D axis.

772. How do you create a 3D line plot?
     → Use `ax.plot3D(x, y, z)`.

773. How do you create a 3D surface plot?
     → Use `ax.plot_surface(X, Y, Z)`.

774. How do you create a 3D wireframe plot?
     → Use `ax.plot_wireframe(X, Y, Z)`.

775. How do you rotate and view 3D plots from different angles?
     → Call `ax.view_init(elev, azim)`.

776. How do you color 3D surfaces based on height or values?
     → Pass `cmap="…" ` to `plot_surface()`.

777. How do you add contours to a 3D surface plot?
     → Use `ax.contour3D(X, Y, Z)`.

778. How do you combine multiple 3D plots in one figure?
     → Create multiple 3D subplots with `fig.add_subplot(..., projection="3d")`.

779. How do you add annotations to 3D points?
     → Use `ax.text(x, y, z, "label")`.

780. How do you optimize 3D plot rendering performance?
     → Use fewer points or lighter methods like wireframes instead of full surfaces.

#### **Heatmaps & Contour Plots**

781. How do you create a basic heatmap using `imshow()`?
     → Use `plt.imshow(data, aspect="auto")`.

782. How do you adjust color limits for heatmaps?
     → Use `plt.clim(min, max)`.

783. How do you add a color bar to a heatmap?
     → Call `plt.colorbar()`.

784. How do you overlay data points on a heatmap?
     → Plot with `plt.scatter()` on top of `imshow()`.

785. How do you create filled contour plots?
     → Use `plt.contourf(X, Y, Z)`.

786. How do you create contour lines without filling?
     → Use `plt.contour(X, Y, Z)`.

787. How do you label contour lines with values?
     → Use `plt.clabel(contour_obj)`.

788. How do you combine contour plots with heatmaps?
     → Draw `imshow()` first, then overlay `contour()`.

789. How do you plot gradients or vector fields?
     → Use `plt.quiver(U, V)` or `plt.streamplot()`.

790. How do you use custom color maps for contour and heatmap plots?
     → Pass `cmap="your_colormap"` to `imshow()` or `contour()`.

#### **Saving & Exporting Plots**

791. How do you save a figure in PNG, JPEG, PDF, or SVG format?
     → Use `plt.savefig("name.png")` (or `.jpg`, `.pdf`, `.svg`).

792. How do you save high-resolution figures using `dpi`?
     → Add `dpi=300` inside `plt.savefig()`.

793. How do you save figures with transparent backgrounds?
     → Use `plt.savefig("file.png", transparent=True)`.

794. How do you save multiple figures programmatically?
     → Loop over figure creation and call `plt.savefig()` each time.

795. How do you export plots for publication quality?
     → Use high DPI, vector formats, and clean styling before saving.

796. How do you save interactive plots as HTML?
     → Use interactive backends like Plotly or mpld3 to export HTML.

797. How do you embed figures in Jupyter notebooks?
     → Use `%matplotlib inline` or `%matplotlib widget`.

798. How do you include vector graphics instead of raster images?
     → Save as `.svg` or `.pdf`.

799. How do you export multiple subplots to a single file?
     → Save the entire figure with `plt.savefig()`.

800. How do you automate figure generation for reports?
     → Write scripts that loop through data and save each plot automatically.

---

### **Batch 9: SciPy & Scientific Tools (Q801–Q900)**

#### **Introduction to SciPy**

801. What is SciPy, and how does it complement NumPy?
     → It adds advanced scientific functions on top of NumPy’s basic array tools.

802. How do you install and import SciPy?
     → Install with `pip install scipy` and import using `import scipy`.

803. What are the main submodules of SciPy?
     → Key ones include `linalg`, `optimize`, `integrate`, `stats`, `interpolate`, and `sparse`.

804. How is SciPy used for scientific and engineering computations?
     → It provides ready-made algorithms for solving math, science, and engineering problems.

805. How do you access linear algebra functions in SciPy?
     → Use `scipy.linalg` for matrix decompositions and solvers.

806. How do you use optimization tools in SciPy?
     → Call functions from `scipy.optimize` like `minimize()`.

807. How do you perform interpolation using SciPy?
     → Use `scipy.interpolate` with tools like `interp1d()`.

808. How do you use SciPy for numerical integration?
     → Use `scipy.integrate` with functions like `quad()` or `odeint()`.

809. How do you handle sparse matrices in SciPy?
     → Use `scipy.sparse` for creating and manipulating sparse matrix types.

810. How do you perform statistical analysis using SciPy?
     → Use `scipy.stats` for distributions, tests, and probability functions.

#### **Linear Algebra**

811. How do you compute eigenvalues and eigenvectors using `scipy.linalg`?
     → Use `scipy.linalg.eig(A)`.

812. How do you perform LU decomposition?
     → Call `scipy.linalg.lu(A)`.

813. How do you perform QR decomposition?
     → Use `scipy.linalg.qr(A)`.

814. How do you perform Cholesky decomposition?
     → Use `scipy.linalg.cholesky(A)`.

815. How do you solve linear systems with `scipy.linalg.solve()`?
     → Call `solve(A, b)`.

816. How do you compute the determinant using SciPy?
     → Use `scipy.linalg.det(A)`.

817. How do you compute the inverse of a matrix?
     → Use `scipy.linalg.inv(A)`.

818. How do you compute matrix norms?
     → Use `scipy.linalg.norm(A)`.

819. How do you perform singular value decomposition (SVD)?
     → Use `scipy.linalg.svd(A)`.

820. How do SciPy linear algebra functions differ from NumPy’s?
     → SciPy’s versions are more robust, optimized, and feature richer algorithms.

#### **Optimization**

821. What is the `scipy.optimize` module used for?
     → It helps you find optimal values that make functions as small, large, or accurate as possible.

822. How do you minimize a scalar function?
     → Use `scipy.optimize.minimize()` with your function and starting point.

823. How do you solve multivariable optimization problems?
     → Pass a multivariable function and an initial guess to `minimize()`.

824. How do you apply constraints in optimization problems?
     → Supply `constraints=` with dictionaries defining equality or inequality rules.

825. How do you use bounds in optimization?
     → Add `bounds=` in `minimize()` to restrict variable ranges.

826. How do you use `minimize_scalar()` for single-variable optimization?
     → Call `minimize_scalar(func)` optionally with a bracket or bounds.

827. How do you solve linear programming problems?
     → Use `scipy.optimize.linprog()` with objective coefficients, constraints, and bounds.

828. How do you solve nonlinear equations using `fsolve()`?
     → Provide the function and an initial guess to `fsolve()` to find roots.

829. How do you perform curve fitting with `curve_fit()`?
     → Give a model function and data to `curve_fit()` to estimate best-fit parameters.

830. How do you select the best optimization algorithm for a problem?
     → Match the algorithm to your function’s smoothness, dimensionality, constraints, and performance needs.

#### **Interpolation**

831. What is interpolation, and why is it used?
     → It’s a way to estimate missing values between known data points to make data smoother or more complete.

832. How do you perform 1D interpolation using `interp1d()`?
     → Create an `interp1d(x, y)` function and call it with new x-values.

833. How do you perform 2D interpolation using `griddata()`?
     → Provide points, values, and a grid to `griddata()` to fill in the missing spots.

834. How do you choose between linear, cubic, and nearest interpolation?
     → Pick linear for balance, cubic for smooth curves, and nearest for rough or categorical data.

835. How do you extrapolate values outside the data range?
     → Enable `fill_value="extrapolate"` in `interp1d()`.

836. How do you interpolate irregularly spaced data?
     → Use `griddata()` or spline tools that accept scattered points.

837. How do you handle multidimensional interpolation?
     → Use methods like `griddata()` or multivariate splines designed for many dimensions.

838. How do you perform spline interpolation using `UnivariateSpline`?
     → Fit `UnivariateSpline(x, y)` and call it like a function.

839. How do you compute derivatives from interpolated functions?
     → Use the spline’s `.derivative()` method or differentiate the interpolation function if supported.

840. How do you evaluate interpolation performance?
     → Compare predicted values with known ones using error metrics like MSE or RMSE.

#### **Numerical Integration**

841. How do you integrate a function numerically using `quad()`?
     → Pass the function and limits to `quad(func, a, b)` to get the integral.

842. How do you perform double or triple integrals using `dblquad()` and `tplquad()`?
     → Provide nested limit functions and the main function to `dblquad()` or `tplquad()`.

843. How do you integrate over arrays using `simps()`?
     → Use `simps(y, x)` to apply Simpson’s rule on array data.

844. How do you use the trapezoidal rule for integration?
     → Call `numpy.trapz(y, x)` to apply the trapezoid method.

845. How do you handle improper integrals?
     → Let `quad()` handle infinite limits by passing `np.inf` or `-np.inf`.

846. How do you integrate functions with singularities?
     → Split the interval around the singularity or rely on `quad()`’s adaptive handling.

847. How do you perform adaptive integration?
     → Use `quad()` since it automatically adapts step sizes.

848. How do you integrate functions defined as discrete data points?
     → Use `trapz()` or `simps()` on the sampled arrays.

849. How do you use cumulative integration (`cumtrapz`)?
     → Call `cumtrapz(y, x)` to get running trapezoidal integrals.

850. How do you evaluate integration accuracy?
     → Compare numerical results to known values or check error estimates returned by the method.

#### **Signal Processing**

851. What is the `scipy.signal` module used for?
     → It’s the toolbox for cleaning, transforming, and analyzing signals so they behave nicely instead of wobbling everywhere.

852. How do you design FIR and IIR filters?
     → Use functions like `firwin()` for FIR and `iirfilter()` or `butter()` for IIR to shape your filter.

853. How do you apply a digital filter to a signal?
     → Feed the signal into `lfilter()` or `filtfilt()` to smooth or modify it.

854. How do you compute the Fourier Transform of a signal?
     → Call `fft()` to turn your time-based signal into frequency goodies.

855. How do you compute the inverse Fourier Transform?
     → Use `ifft()` to convert frequency data back to the time domain.

856. How do you perform convolution of two signals?
     → Use `convolve()` to mix the two signals together mathematically.

857. How do you perform correlation of two signals?
     → Use `correlate()` to see how similar they are when shifted around.

858. How do you perform decimation and resampling?
     → Use `decimate()` or `resample()` to shrink or reshape the number of samples.

859. How do you detect peaks in a signal?
     → Run `find_peaks()` to spot the high points like a tiny mountain detector.

860. How do you perform windowing for spectral analysis?
     → Multiply your signal by a window from `scipy.signal.windows` to reduce edge jumpiness.

#### **Sparse Matrices**

861. What are sparse matrices, and why are they used?
     → They’re matrices mostly filled with zeros, used to save memory and speed up calculations.

862. How do you create a CSR (Compressed Sparse Row) matrix?
     → Use `csr_matrix((data, indices, indptr))` or convert with `.tocsr()`.

863. How do you create a CSC (Compressed Sparse Column) matrix?
     → Use `csc_matrix((data, indices, indptr))` or convert with `.tocsc()`.

864. How do you create a COO (Coordinate) sparse matrix?
     → Use `coo_matrix((data, (row, col)))`.

865. How do you convert between sparse and dense matrices?
     → Call `.toarray()` or `.A` for dense, or use `.tocsr()`, `.tocsc()`, `.coo()` for sparse formats.

866. How do you perform arithmetic operations on sparse matrices?
     → Use normal operators like `+`, `-`, `*` which work efficiently in sparse format.

867. How do you perform matrix multiplication with sparse matrices?
     → Use the `@` operator or `.dot()`.

868. How do you solve linear systems with sparse matrices?
     → Apply `spsolve()` or iterative methods like `cg()`.

869. How do you store sparse matrices efficiently?
     → Pick formats like CSR or CSC that compress zeros and keep structure compact.

870. How do sparse matrices improve computational performance?
     → They reduce memory use and skip needless operations on zeros, making everything faster.

#### **Statistical Analysis**

871. How do you compute descriptive statistics using `scipy.stats`?
     → Use functions like `describe()` to get a bundle of summary stats in one go.

872. How do you compute mean, median, and variance?
     → Call `stats.tmean()`, `stats.tmedian()`, or `stats.tvar()`.

873. How do you compute skewness and kurtosis?
     → Use `skew()` and `kurtosis()` for shape-related measures.

874. How do you perform hypothesis testing using `ttest_ind()`?
     → Provide two independent samples to `ttest_ind(sample1, sample2)`.

875. How do you perform paired t-tests using `ttest_rel()`?
     → Pass two related samples to `ttest_rel(a, b)`.

876. How do you perform ANOVA using `f_oneway()`?
     → Give your groups to `f_oneway(group1, group2, ...)`.

877. How do you compute correlation coefficients?
     → Use `pearsonr()`, `spearmanr()`, or `kendalltau()` depending on the type.

878. How do you compute ranks and rank correlations?
     → Use `rankdata()` for ranks and `spearmanr()` or `kendalltau()` for correlation.

879. How do you generate random samples from distributions?
     → Call distribution methods like `stats.norm.rvs(size=n)`.

880. How do you test for normality in data?
     → Use tests like `shapiro()` or `normaltest()` to check distribution shape.

#### **Probability Distributions**

881. How do you use continuous probability distributions in SciPy?
     → Call distributions like `stats.norm`, then use their methods such as `pdf`, `cdf`, or `rvs`.

882. How do you use discrete probability distributions?
     → Use classes like `stats.binom` or `stats.poisson` with the same method pattern.

883. How do you compute PDF and CDF values?
     → Use `.pdf(x)` for density and `.cdf(x)` for cumulative probability.

884. How do you compute percent point function (PPF)?
     → Call `.ppf(q)` to get the value at a given percentile.

885. How do you compute survival functions (SF)?
     → Use `.sf(x)` to get the probability of being above x.

886. How do you generate random samples from a given distribution?
     → Call `.rvs(size=n)` on the distribution object.

887. How do you fit a distribution to data?
     → Use `.fit(data)` to estimate parameters.

888. How do you compute moments of a distribution?
     → Call methods like `.moment(k)` or use `.mean()`, `.var()`, `.skew()`, `.kurt()`.

889. How do you perform statistical tests for distribution fit?
     → Use tests like `kstest()` or `chisquare()`.

890. How do you compare two distributions statistically?
     → Apply tests such as `ks_2samp()` or `mannwhitneyu()`.

#### **Curve Fitting & Regression**

891. How do you fit a curve using `curve_fit()`?
     → Provide a model function and data to `curve_fit()` to get best-fit parameters.

892. How do you choose a fitting function?
     → Pick a formula that reflects the pattern you expect in the data.

893. How do you extract optimal parameters and covariance?
     → `curve_fit()` returns `popt` for parameters and `pcov` for their covariance.

894. How do you handle nonlinear curve fitting?
     → Use a nonlinear model and let `curve_fit()` iterate from an initial guess.

895. How do you perform weighted curve fitting?
     → Pass `sigma=` to `curve_fit()` so points with lower variance count more.

896. How do you evaluate goodness of fit?
     → Check residuals, R², or compare errors between model and actual data.

897. How do you handle outliers in curve fitting?
     → Filter them, use robust methods, or down-weight them with larger `sigma`.

898. How do you fit multiple datasets simultaneously?
     → Combine them into one objective function or merge arrays before fitting.

899. How do you use polynomial fitting with SciPy?
     → Use `numpy.polyfit()` or wrap a polynomial model for `curve_fit()`.

900. How do you visualize fitted curves alongside data?
     → Plot raw points and overlay the model’s predicted curve.

---

### **Batch 10: Data Analysis Ecosystems & Deployment (Q901–Q1000)**

#### **Data Wrangling Pipelines**

901. How do you combine Pandas and NumPy efficiently in a data pipeline?
     → Use Pandas for structure and NumPy for fast math, swapping between them with `.values` or `to_numpy()`.

902. How do you handle missing data in a pipeline workflow?
     → Apply `fillna()`, `dropna()`, or interpolation early in the chain.

903. How do you chain multiple transformations using method chaining?
     → Link operations with dots and wrap steps in parentheses for clarity.

904. How do you perform feature engineering in a data pipeline?
     → Create new columns using vectorized NumPy or Pandas expressions.

905. How do you normalize or scale features in a pipeline?
     → Apply operations like `(col - col.mean()) / col.std()` or use sklearn scalers if needed.

906. How do you apply custom functions in a pipeline?
     → Use `.apply()`, `.assign()`, or vectorized NumPy functions.

907. How do you handle categorical variables in a pipeline?
     → Convert them using `get_dummies()` or Pandas categoricals.

908. How do you merge multiple datasets in a pipeline?
     → Use `merge()`, `concat()`, or `join()` in the flow.

909. How do you filter and select relevant columns efficiently?
     → Use boolean indexing, column lists, or `filter()`.

910. How do you maintain reproducibility in a data pipeline?
     → Fix random seeds, version data sources, and store transformation steps clearly.

#### **Visualization Integration**

911. How do you integrate Matplotlib plots with Pandas data?
     → Call `.plot()` directly on DataFrames since they hook into Matplotlib automatically.

912. How do you use Seaborn with Pandas DataFrames?
     → Pass column names directly to Seaborn functions because they read DataFrames natively.

913. How do you use Plotly for interactive visualizations with Pandas?
     → Feed DataFrames into Plotly Express functions like `px.line(df, x, y)`.

914. How do you combine multiple visualizations in a single workflow?
     → Create several plots step-by-step and stack or sequence them in one script or notebook cell.

915. How do you create dashboards with Python visualization libraries?
     → Use tools like Plotly Dash or Streamlit to assemble plots into interactive layouts.

916. How do you update visualizations dynamically in a notebook?
     → Re-run the cell or use widgets like ipywidgets to refresh plots on the fly.

917. How do you export visualizations for reporting?
     → Save them using `.savefig()` or Plotly’s export functions.

918. How do you overlay multiple plot types in one figure?
     → Plot them on the same axes or add layers before showing the figure.

919. How do you customize color palettes for integrated visualizations?
     → Pick palettes from Seaborn or define custom lists and pass them into plotting functions.

920. How do you visualize grouped or aggregated data effectively?
     → Group with Pandas, then chart the results using bars, lines, or boxplots.

#### **Workflow Tools**

921. What is Jupyter Notebook, and why is it used for data analysis?
     → It’s an interactive workspace where you mix code, text, and visuals to explore data easily.

922. How do you install and launch Jupyter Notebook?
     → Install via pip or Anaconda and run `jupyter notebook` in a terminal.

923. How do you create and run cells in Jupyter Notebook?
     → Add cells with the toolbar and run them using Shift+Enter.

924. How do you use Markdown and code cells effectively?
     → Use code cells for execution and Markdown cells for explanations, headings, and notes.

925. How do you use IPython magic commands for workflow efficiency?
     → Prefix commands with `%` or `%%` to access shortcuts like `%ls`, `%timeit`, or `%%bash`.

926. How do you profile code execution in Jupyter using `%timeit`?
     → Write `%timeit your_code` to get fast performance estimates.

927. How do you export notebooks to HTML or PDF?
     → Use `File → Download as` or run `jupyter nbconvert`.

928. How do you organize large projects using notebooks?
     → Split work into separate notebooks and keep them in tidy folders.

929. How do you use Jupyter Lab for advanced workflows?
     → Launch it with `jupyter lab` to get tabs, split views, and better file management.

930. How do you integrate notebooks with version control systems?
     → Store `.ipynb` files in Git and use tools like `nbdime` for cleaner diffs.

#### **Deployment & Automation**

931. How do you convert a Python script into a scheduled task?
     → Use system schedulers like cron or Task Scheduler to run your script at set times.

932. How do you use cron jobs to automate Python scripts?
     → Add a line in `crontab -e` specifying the schedule and Python command.

933. How do you use Windows Task Scheduler for Python scripts?
     → Create a new task, point it to your Python executable and script, and set a trigger.

934. How do you automate data ingestion pipelines?
     → Chain scripts or tools that fetch, clean, and store data on a schedule.

935. How do you deploy Python scripts as web services?
     → Wrap them with frameworks like Flask or FastAPI and run on a server.

936. How do you use Airflow for workflow scheduling and orchestration?
     → Define tasks as DAGs and let Airflow manage dependencies and timing.

937. How do you monitor deployed pipelines for failures?
     → Use logs, alerts, and monitoring tools to catch issues early.

938. How do you handle logging and error reporting in deployed scripts?
     → Use Python’s `logging` module and send errors to files or alert systems.

939. How do you create reusable scripts for different datasets?
     → Parameterize file paths and settings so the same script adapts easily.

940. How do you manage dependencies in deployed environments?
     → Use virtual environments or tools like `requirements.txt` to lock versions.

#### **Performance Tuning**

941. How do you profile Python code to find bottlenecks?
     → Use profiling tools to spot slow lines or functions in your code.

942. How do you use `cProfile` for performance profiling?
     → Run `python -m cProfile script.py` or import it and profile specific functions.

943. How do you optimize loops with vectorized operations?
     → Replace Python loops with NumPy operations that run in optimized C code.

944. How do you reduce memory usage with appropriate data types?
     → Pick smaller dtypes like `float32` or `int16` when precision allows.

945. How do you optimize Pandas and NumPy operations for large datasets?
     → Use vectorization, avoid Python loops, and work in chunks if needed.

946. How do you parallelize computations using `multiprocessing`?
     → Spawn multiple processes with `Pool` or `Process` to split heavy work.

947. How do you use Dask for out-of-core and parallel computing?
     → Replace NumPy or Pandas with Dask equivalents and let Dask schedule tasks.

948. How do you balance memory and CPU performance in pipelines?
     → Tune batch sizes, pick efficient dtypes, and avoid unnecessary copies.

949. How do you cache intermediate results for efficiency?
     → Save results to disk or memory so repeated steps don’t recompute.

950. How do you benchmark code before and after optimizations?
     → Time sections with `%timeit` or `time` to compare improvements.

#### **Advanced Data Handling**

951. How do you handle large CSV files efficiently?
     → Read in chunks, use efficient dtypes, or switch to faster formats.

952. How do you read and write Parquet files with Pandas?
     → Use `read_parquet()` and `to_parquet()` for fast, compressed columnar storage.

953. How do you use HDF5 files for large datasets?
     → Store and access data with `HDFStore` for quick, structured retrieval.

954. How do you integrate SQL databases with Pandas?
     → Use `read_sql()` and `to_sql()` with a database connection.

955. How do you optimize database queries for Pandas workflows?
     → Filter, aggregate, and limit data inside SQL before loading it.

956. How do you handle streaming data in Python pipelines?
     → Process data as it arrives using generators or streaming libraries.

957. How do you combine batch and streaming data processing?
     → Merge periodic batch loads with real-time event streams in the same pipeline.

958. How do you manage versioning of large datasets?
     → Use tools like DVC or store timestamped snapshots.

959. How do you perform incremental updates to large datasets?
     → Append only new records and track offsets or timestamps.

960. How do you maintain data integrity across multiple sources?
     → Validate schemas, enforce constraints, and cross-check data during ingestion.

#### **Visualization & Reporting**

961. How do you create automated reports using Python scripts?
     → Generate text, tables, and plots in scripts and export them as files like HTML or PDF.

962. How do you export Pandas DataFrames as tables in reports?
     → Convert them to HTML, LaTeX, or images and embed them in the report.

963. How do you embed interactive visualizations in reports?
     → Use Plotly or Bokeh and export the interactive HTML.

964. How do you schedule report generation automatically?
     → Trigger the script with cron or Task Scheduler at regular intervals.

965. How do you format visualizations for presentation?
     → Clean up labels, titles, and layouts so the plots look polished.

966. How do you integrate Matplotlib, Seaborn, and Plotly for consistent reports?
     → Use shared color themes and styles across all three libraries.

967. How do you create summary dashboards using Python?
     → Build dashboards with libraries like Dash or Streamlit.

968. How do you visualize time series trends over long periods?
     → Plot smoothed lines, rolling stats, or zoomable interactive charts.

969. How do you create comparative visualizations for multiple datasets?
     → Overlay or arrange plots side-by-side to show differences clearly.

970. How do you ensure reproducibility of visual reports?
     → Fix random seeds, version data, and keep scripts consistent.

#### **Parallelization & Scalability**

971. How do you use `joblib` for parallel computing in Python?
     → Use `Parallel` and `delayed()` to run loops across multiple cores.

972. How do you use `concurrent.futures` for parallel tasks?
     → Create a `ThreadPoolExecutor` or `ProcessPoolExecutor` and submit tasks.

973. How do you distribute computations across multiple cores?
     → Use multiprocessing, joblib, or Dask to split work into parallel chunks.

974. How do you handle large datasets with limited memory?
     → Stream, chunk, or process data lazily instead of loading it all at once.

975. How do you process data in chunks to improve scalability?
     → Read or compute in fixed-size blocks and handle each block independently.

976. How do you use Dask DataFrames for parallel Pandas workflows?
     → Replace Pandas with Dask DataFrames and let Dask split work across cores.

977. How do you balance CPU and memory usage in scalable pipelines?
     → Tune chunk sizes, limit parallelism, and pick efficient data types.

978. How do you avoid race conditions in parallel processing?
     → Use locks, queues, or avoid shared mutable state entirely.

979. How do you profile parallel workflows for efficiency?
     → Measure task times, worker usage, and bottlenecks with profiling tools.

980. How do you monitor distributed computations in Python?
     → Use Dask’s dashboard or logging from your parallel framework.

#### **Data Validation & Quality**

981. How do you validate input data in pipelines?
     → Apply checks on types, ranges, and formats before processing.

982. How do you check for missing or null values?
     → Use `isnull()` or `isna()` and count or filter them.

983. How do you enforce data type consistency?
     → Convert columns with `astype()` or schema rules.

984. How do you detect duplicates in large datasets?
     → Use `duplicated()` and filter or drop them.

985. How do you handle inconsistent categorical values?
     → Normalize them with mapping, cleaning, or category lists.

986. How do you validate ranges and limits of numerical data?
     → Check with boolean conditions like `(col >= low) & (col <= high)`.

987. How do you log validation errors efficiently?
     → Write issues to structured logs or error files as they appear.

988. How do you perform automated sanity checks on datasets?
     → Run predefined rules that confirm shape, ranges, and distribution patterns.

989. How do you generate reports on data quality?
     → Summarize validation results into tables or dashboards.

990. How do you integrate data validation with deployment pipelines?
     → Make validation a required step before running main tasks.

#### **Scripting & Automation Best Practices**

991. How do you structure Python scripts for reusability?
     → Break logic into functions and keep reusable parts in separate modules.

992. How do you modularize code for readability?
     → Group related functions into files and keep each module focused.

993. How do you use configuration files to manage parameters?
     → Store settings in YAML/JSON/INI and load them at runtime.

994. How do you handle logging and exception handling in scripts?
     → Use the `logging` module and wrap risky code in try–except blocks.

995. How do you create command-line interfaces for scripts?
     → Use `argparse` to parse options and arguments.

996. How do you test scripts before deployment?
     → Write unit tests and run them with tools like pytest.

997. How do you automate repetitive data tasks efficiently?
     → Wrap tasks in functions and run them on schedules or loops.

998. How do you integrate version control in deployed scripts?
     → Track scripts with Git and tag stable releases.

999. How do you schedule scripts for batch processing?
     → Use cron, Task Scheduler, or workflow tools to run them periodically.

1000. How do you ensure maintainability and scalability of data pipelines?
      → Keep code clean, document flows, and split big tasks into modular stages.

---
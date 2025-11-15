# **Java Programming Interview Questions**


---

## 🧩 **Batch 1 — Core Java Basics to Intermediate (Q1–Q100)**

### **Java Basics**

1. What is Java, and why is it platform-independent?
   → Java runs on the JVM, which makes the same bytecode work on any machine—like magic slippers for code.

2. Explain the difference between JDK, JRE, and JVM.
   → JDK is the full toolbox, JRE is the running environment, and JVM is the engine that actually runs your bytecode.

3. What are the main features of Java?
   → It’s object-oriented, portable, secure, and has garbage collection to clean up your mess.

4. What is the difference between Java and other programming languages like C++?
   → Java removes tricky stuff like pointers and manual memory management, making life a little less stressful.

5. What is bytecode in Java?
   → It’s the intermediate code the JVM understands—a universal language for all Java programs.

6. What is the Java Virtual Machine (JVM)?
   → It’s the software engine that runs Java bytecode and keeps things platform-independent.

7. What are the different types of memory areas allocated by JVM?
   → Heap, Stack, Method Area, PC Registers, and Native Method Stack—like rooms in a tiny code hotel.

8. What is the role of the Just-In-Time (JIT) compiler in Java?
   → It turns hot bytecode into machine code on the fly to make your program zoom faster.

9. What is the difference between a compiler and an interpreter?
   → A compiler translates everything at once; an interpreter reads it line by line like a storyteller.

10. What is the significance of the `main()` method in Java?
    → It’s the official entry door where every Java program begins its journey.

11. Why is Java considered an object-oriented language?
    → Because it focuses on objects and classes, making everything feel modular and neat.

12. What are objects and classes in Java?
    → Classes are blueprints, and objects are the actual things built from those blueprints.

13. What is the difference between a class and an object?
    → A class is the idea; an object is the real thing created from that idea.

14. What is a constructor in Java?
    → It’s a special method that sets up an object when it’s born.

15. What is the default constructor?
    → It’s the empty constructor Java auto-creates when you don’t provide one.

16. Can a constructor be private?
    → Yes, usually for patterns like singletons to keep object creation controlled.

17. Can a constructor return a value?
    → No—constructors don’t return anything, not even a sneaky void.

18. What is the difference between `this` and `super` keywords?
    → `this` points to the current class, while `super` reaches up to the parent class.

19. What is method overloading?
    → Same method name, different parameters—a friendly multitasker.

20. What is method overriding?
    → When a subclass rewrites a parent method to give it a fresh personality.


### **Data Types & Variables**

21. What are primitive data types in Java?
    → They are the basic built-in types like int, char, double, and boolean—Java’s tiny building blocks.

22. What is the difference between primitive and reference data types?
    → Primitives store actual values, while references store addresses pointing to objects.

23. What is type casting in Java?
    → It’s converting one data type into another—like fitting data into a new costume.

24. What is the difference between implicit and explicit casting?
    → Implicit is automatic widening, explicit is manual narrowing.

25. What are wrapper classes in Java?
    → Object versions of primitive types, like Integer for int and Double for double.

26. Why are wrapper classes needed?
    → They let primitives work in object-only places like collections.

27. What is autoboxing and unboxing?
    → Autoboxing converts primitives to wrappers; unboxing converts them back.

28. What is the default value of different data types in Java?
    → Primitives get zeros or false, references get null.

29. What is the difference between `==` and `.equals()`?
    → `==` checks memory addresses, `.equals()` checks actual content.

30. What is the difference between `String`, `StringBuilder`, and `StringBuffer`?
    → String is immutable, StringBuilder is mutable and fast, StringBuffer is mutable and thread-safe.

### **Control Flow Statements**

31. What are the different control statements in Java?
    → Selection (if, switch), looping (for, while, do-while), and jump statements (break, continue, return).

32. How does the `switch` statement work?
    → It checks a value against multiple cases and runs the matching block.

33. Can a `switch` statement use a `String`?
    → Yes, from Java 7 onward.

34. What is the difference between `break` and `continue`?
    → `Break` stops the loop entirely, `continue` skips to the next iteration.

35. What is the difference between `while` and `do-while` loops?
    → `While` checks first, `do-while` runs once before checking.

36. Can a `for` loop be infinite?
    → Yes—just leave out the condition.

37. What is the enhanced for loop (for-each loop)?
    → A simplified loop used to traverse arrays and collections.

38. What is the difference between `return`, `break`, and `continue`?
    → `Return` exits the method, `break` exits a loop or switch, `continue` skips to the next loop cycle.

39. Can we use multiple `return` statements in a method?
    → Yes, as long as the flow logically allows it.

40. What happens if you forget a `break` in a `switch` case?
    → Execution “falls through” into the next case.


### **OOPs Concepts**

41. What are the four pillars of Object-Oriented Programming?
    → Inheritance, Polymorphism, Encapsulation, and Abstraction—the fantastic four of OOP.

42. What is inheritance?
    → A way for one class to reuse and extend another class’s features.

43. What is polymorphism?
    → One action behaving differently depending on the object using it.

44. What is encapsulation?
    → Wrapping data and methods together and guarding them with access control.

45. What is abstraction?
    → Showing only the important stuff and hiding all the messy details.

46. What is the difference between abstraction and encapsulation?
    → Abstraction hides complexity, encapsulation protects access.

47. What is an abstract class?
    → A class you can’t fully use until a subclass fills in the missing pieces.

48. What is an interface?
    → A contract listing methods that classes must implement.

49. Can an interface have method implementations?
    → Yes—default and static methods can have bodies.

50. What is the difference between an abstract class and an interface?
    → Abstract classes have partial implementation; interfaces mostly define rules.

51. Can we create an object of an abstract class?
    → Nope—it's like trying to use a sketch as a finished machine.

52. Can a class implement multiple interfaces?
    → Yes—Java happily allows it.

53. What is multiple inheritance?
    → A class having more than one parent.

54. Does Java support multiple inheritance?
    → Not with classes, to avoid chaos.

55. How can we achieve multiple inheritance in Java?
    → By implementing multiple interfaces.

56. What is a static method?
    → A method that belongs to the class, not the object.

57. Can we override a static method?
    → No—static methods can only be hidden, not overridden.

58. What is a final variable, method, or class?
    → Final variable can’t change, final method can’t be overridden, final class can’t be extended.

59. What is the difference between `final`, `finally`, and `finalize()`?
    → `final` restricts, `finally` always runs, `finalize()` was for cleanup but is now basically retired.

60. What is the `super` keyword used for?
    → To call parent class methods, constructors, or access its hidden members.


### **Exception Handling**

61. What is exception handling in Java?
    → A way to manage runtime problems without crashing the program.

62. What are checked and unchecked exceptions?
    → Checked must be handled at compile time; unchecked appear at runtime.

63. What is the `try-catch` block used for?
    → To wrap risky code and handle errors safely.

64. Can we have multiple `catch` blocks?
    → Yes—each can handle a different exception type.

65. What is the `finally` block?
    → A block that runs no matter what, usually for cleanup.

66. Can `finally` block be skipped?
    → Only if the JVM shuts down unexpectedly.

67. What is the `throw` keyword used for?
    → To manually create and send out an exception.

68. What is the `throws` keyword used for?
    → To declare that a method might toss certain exceptions.

69. Can we have a `try` block without `catch`?
    → Yes, if you use a `finally` block.

70. What is the difference between `throw` and `throws`?
    → `throw` launches an exception; `throws` announces potential exceptions.

71. What happens when an exception is not handled?
    → The program terminates and the JVM shows an error message.

72. Can we create custom exceptions in Java?
    → Yes—Java lets you craft your own exception types.

73. How do we create a custom exception?
    → Make a class that extends `Exception` or `RuntimeException`.

74. What is the root class of all exceptions in Java?
    → `Throwable` sits at the very top.

75. What is the difference between `Error` and `Exception`?
    → Errors are severe system issues; exceptions are recoverable program issues.


### **Collections Framework (Intro)**

76. What is the Java Collections Framework?
    → A set of ready-made data structures and algorithms to manage groups of objects.

77. What is the difference between a Collection and a Map?
    → Collection stores elements; Map stores key-value pairs.

78. What are the main interfaces in the Java Collections Framework?
    → List, Set, Queue, Deque, and Map.

79. What is the difference between List, Set, and Map?
    → List allows duplicates, Set avoids duplicates, Map stores key-value pairs.

80. What is the difference between ArrayList and LinkedList?
    → ArrayList is fast at access, LinkedList is fast at insert/delete.

81. What is the difference between HashSet and TreeSet?
    → HashSet is fast and unordered; TreeSet is sorted and slower.

82. What is the difference between HashMap and TreeMap?
    → HashMap is fast and unordered; TreeMap is sorted by keys.

83. What is the difference between Iterator and ListIterator?
    → Iterator moves forward; ListIterator moves both ways and supports more operations.

84. What is the difference between fail-fast and fail-safe iterators?
    → Fail-fast crash on modification; fail-safe work on a copy.

85. What is the difference between HashMap and Hashtable?
    → HashMap is non-synchronized and faster; Hashtable is synchronized and older.

86. What is the load factor in HashMap?
    → A threshold that triggers resizing, usually 0.75.

87. What happens if two keys in a HashMap have the same hashcode?
    → They land in the same bucket and form a chain or tree.

88. What is the internal structure of a HashMap?
    → An array of buckets using linked lists or balanced trees.

89. What is the difference between `Array` and `ArrayList`?
    → Array is fixed-size; ArrayList grows dynamically.

90. Can we store null values in a HashMap?
    → Yes—one null key and multiple null values.


### **Multithreading (Intro)**

91. What is a thread in Java?
    → A lightweight unit of execution that runs tasks concurrently.

92. How do you create a thread in Java?
    → Either extend `Thread` or implement `Runnable` and pass it to a `Thread` object.

93. What is the difference between extending `Thread` and implementing `Runnable`?
    → Extending limits inheritance; implementing is flexible and preferred.

94. What is thread synchronization?
    → A way to prevent multiple threads from messing with shared data at the same time.

95. What is a synchronized block?
    → A protected section of code that only one thread can enter at once.

96. What is a deadlock?
    → When threads wait on each other forever like stubborn statues.

97. What is the difference between `wait()` and `sleep()`?
    → `wait()` releases the lock; `sleep()` doesn’t.

98. What is the difference between `notify()` and `notifyAll()`?
    → `notify()` wakes one waiting thread; `notifyAll()` wakes them all.

99. What is thread priority?
    → A hint to the scheduler about how important a thread is.

100. What is the difference between a process and a thread?
     → A process is a full program; threads are smaller tasks inside it sharing resources.

---

## 🧠 **Batch 2 — Advanced Core Java & JVM Internals (Q101–Q200)**

### **JVM & Memory Management**

101. What are the different components of JVM architecture?
     → ClassLoader, Runtime Data Areas, Execution Engine, and Native Interface—basically JVM’s internal city map.

102. What is the ClassLoader in Java?
     → A module that loads classes into memory like a librarian fetching books.

103. What are the types of ClassLoaders?
     → Bootstrap, Extension, and Application ClassLoaders.

104. What is the purpose of the Bootstrap ClassLoader?
     → To load core Java classes—it's the boss loader.

105. What is the Method Area in JVM memory?
     → A zone storing class-level data like methods, constants, and bytecode.

106. What is the difference between heap and stack memory?
     → Heap stores objects; stack stores method calls and local variables.

107. What is garbage collection in Java?
     → Automatic cleanup of unused objects—Java’s own housekeeping robot.

108. How does the garbage collector know which objects to delete?
     → It checks reachability—if nobody points to it, it’s out.

109. What are strong, weak, soft, and phantom references?
     → Different "strengths" of how tightly objects are held for GC decisions.

110. What are memory leaks in Java, and how do they occur?
     → When unused objects stay referenced, hogging memory like unreturned library books.

111. How do you trigger garbage collection manually?
     → Call `System.gc()`, but it’s just a polite request, not a command.

112. What is the `finalize()` method used for?
     → A last-chance cleanup callback before GC—like a farewell whisper.

113. Is it a good idea to rely on `finalize()`? Why or why not?
     → No, it’s unpredictable and deprecated.

114. What is the difference between minor GC and major GC?
     → Minor GC cleans young generation; major GC cleans old generation and is heavier.

115. What is the Metaspace in Java 8 and above?
     → A native-memory area storing class metadata, replacing PermGen.

116. How do you analyze memory usage in a Java application?
     → Use profiling tools, logs, and heap dumps.

117. What tools can be used for profiling Java memory?
     → VisualVM, JProfiler, YourKit, and Java Mission Control.

118. What are OutOfMemoryError and StackOverflowError?
     → OOM means memory ran out; StackOverflow means recursion went wild.

119. How can you prevent memory leaks in Java?
     → Remove unwanted references, close resources, and avoid static hoarding.

120. What is the difference between JVM tuning and profiling?
     → Tuning adjusts performance settings; profiling inspects behavior to find issues.


### **Advanced OOP & Design**

121. What is composition in Java?
     → Using one class inside another to build complex objects.

122. How does composition differ from inheritance?
     → Composition uses “has-a”; inheritance uses “is-a”.

123. What are access modifiers in Java?
     → Keywords controlling visibility: public, private, protected, and default.

124. What is the default access modifier?
     → Package-private—visible only within the same package.

125. What are nested classes in Java?
     → Classes defined inside another class.

126. What are static nested classes?
     → Nested classes that don’t need an outer class object to be used.

127. What is an anonymous inner class?
     → A nameless class created for quick one-time use.

128. What are local inner classes?
     → Classes defined inside a method.

129. What is the purpose of `this` and `super` in constructor chaining?
     → `this` calls another constructor in the same class; `super` calls a parent constructor.

130. What is a copy constructor?
     → A constructor that makes a new object by copying another one.

131. How can you implement a copy constructor in Java?
     → Create a constructor that accepts the same class type and copies its fields.

132. What is the clone() method in Java?
     → A method to duplicate objects using the Cloneable mechanism.

133. What is shallow copy vs deep copy?
     → Shallow copies references; deep copy clones everything inside too.

134. How can you achieve deep copy in Java?
     → Clone fields manually, use copy constructors, or use serialization.

135. Why should we override the `equals()` and `hashCode()` methods together?
     → To keep consistent behavior in hashed collections.

136. What is the `toString()` method used for?
     → To return a readable string version of an object.

137. Can we override `equals()` without overriding `hashCode()`?
     → Yes, but it breaks hashed collections.

138. What is immutability in Java?
     → When an object cannot be changed after creation.

139. How do you create an immutable class?
     → Make fields final, private, no setters, and return copies of mutable fields.

140. What is the `Object` class in Java, and what are its key methods?
     → The root class of all Java classes containing methods like equals, hashCode, toString, clone, and finalize.


### **Generics**

141. What are generics in Java?
     → A way to write flexible, type-safe classes and methods.

142. Why were generics introduced?
     → To prevent type errors and reduce casting headaches.

143. What are the advantages of using generics?
     → Type safety, cleaner code, and reusable components.

144. What is type erasure in Java generics?
     → The process where generic types vanish at runtime.

145. Can generics be used with primitive types?
     → No—only reference types.

146. What are bounded type parameters?
     → Type parameters restricted using `extends` or `super`.

147. What is the difference between `<T>`, `<T extends Class>`, and `<T super Class>`?
     → `<T>` is generic, `extends` is upper-bound, `super` is lower-bound.

148. What are wildcards in generics?
     → Question marks that allow flexible unknown types.

149. What is the difference between `List<?>`, `List<Object>`, and `List<? extends Object>`?
     → `List<?>` accepts anything read-only, `List<Object>` needs exact Object, `? extends Object` allows reading from any subtype.

150. What is a generic method?
     → A method with its own type parameter.

151. Can a generic class implement a non-generic interface?
     → Yes—no issues there.

152. Can you create an array of generics? Why or why not?
     → No—runtime type erasure breaks array type safety.

153. What is type inference?
     → The compiler guessing the generic type automatically.

154. What are the limitations of generics in Java?
     → No primitives, no arrays, and erased types.

155. Can you overload methods using generics?
     → Yes, but type erasure can cause conflicts.

156. How do you create a generic class?
     → Add `<T>` after the class name and use T as a placeholder.

157. What is the diamond operator (`<>`) in Java?
     → A shorthand letting the compiler infer generic types.

158. What is the difference between compile-time and runtime type checking in generics?
     → Generics enforce type safety at compile-time; runtime sees erased types.

159. What happens to generics during runtime?
     → They’re erased and operate as raw types.

160. How do generics improve code reusability?
     → By letting one structure handle many data types safely.


### **Collections Framework (Deep Dive)**

161. How does an ArrayList work internally?
     → It uses a dynamic array that grows as needed.

162. What happens when an ArrayList reaches its capacity?
     → It expands by creating a bigger array and copying elements.

163. How does a LinkedList work internally?
     → It uses nodes linked together with next/previous pointers.

164. What is the internal data structure of a HashSet?
     → A HashMap underneath, storing elements as keys.

165. How does HashMap resolve hash collisions?
     → Using linked lists or balanced trees in buckets.

166. What is rehashing in HashMap?
     → Recomputing bucket positions when resizing.

167. How does the equals() and hashCode() contract affect HashMap behavior?
     → Proper hashing ensures correct key lookup and storage.

168. What is the internal structure of a ConcurrentHashMap?
     → A segmented, lock-efficient hash table.

169. What is the difference between ConcurrentHashMap and synchronizedMap()?
     → ConcurrentHashMap uses fine-grained locking; synchronizedMap locks the whole map.

170. What is CopyOnWriteArrayList and how does it work?
     → A thread-safe list that copies the array on each write.

171. What is the difference between Hashtable and ConcurrentHashMap?
     → Hashtable locks everything; ConcurrentHashMap locks only parts.

172. What is the purpose of WeakHashMap?
     → To allow keys to be garbage collected when weakly referenced.

173. What is the difference between TreeMap and LinkedHashMap?
     → TreeMap is sorted; LinkedHashMap keeps insertion order.

174. How does LinkedHashMap maintain insertion order?
     → Through a doubly linked list of entries.

175. How can you make a Collection thread-safe?
     → Use `Collections.synchronizedXXX()` or concurrent classes.

176. What is the difference between synchronized collection and concurrent collection?
     → Synchronized is fully locked; concurrent allows safe parallel access.

177. What is Enumeration and Iterator?
     → Enumeration is the old cursor; Iterator is modern and supports removal.

178. What is a BlockingQueue?
     → A queue that waits when adding/removing until space/data appears.

179. What is the difference between PriorityQueue and ArrayDeque?
     → PriorityQueue orders elements by priority; ArrayDeque is a fast double-ended queue.

180. How do you sort a List of objects in Java?
     → Use `Collections.sort()` or `list.sort()` with a Comparator.


### **Functional Programming (Java 8 and Beyond)**

181. What are lambda expressions in Java?
     → Short functions written in a compact, arrow-style format.

182. What is a functional interface?
     → An interface with exactly one abstract method.

183. Give examples of built-in functional interfaces in Java.
     → Runnable, Predicate, Function, Supplier, Consumer.

184. What is the difference between a lambda expression and an anonymous class?
     → Lambdas are shorter and focus on behavior, not type.

185. What is the `Predicate` interface used for?
     → Testing a condition and returning true/false.

186. What is the `Function` interface used for?
     → Converting one value into another.

187. What is the `Supplier` interface used for?
     → Providing values without any input.

188. What is the `Consumer` interface used for?
     → Accepting a value and performing an action.

189. What is a method reference?
     → A shorthand way to refer to an existing method.

190. What is the syntax for constructor references?
     → ClassName::new.

191. What are streams in Java?
     → Pipelines for processing data in a functional style.

192. What is the difference between intermediate and terminal operations in streams?
     → Intermediate steps transform data; terminal steps produce results.

193. What are some examples of intermediate operations?
     → map, filter, sorted, flatMap, distinct.

194. What are some examples of terminal operations?
     → collect, forEach, reduce, count, findFirst.

195. What is the difference between `map()` and `flatMap()`?
     → map transforms elements; flatMap flattens nested structures too.

196. What is the difference between sequential and parallel streams?
     → Sequential runs in one thread; parallel uses multiple threads.

197. What is lazy evaluation in Java streams?
     → Operations run only when a terminal operation is called.

198. How can you filter and sort using streams?
     → Use filter() for conditions and sorted() for ordering.

199. What are Optional objects in Java?
     → Containers that may or may not hold a value.

200. What is the purpose of the `Optional` class, and how do you use it?
     → To avoid null checks—use methods like of(), empty(), get(), orElse().


---

## ⚙️ **Batch 3 — Advanced Concurrency & Multithreading (Q201–Q300)**

### **Threads & Life Cycle**

201. What are the different states of a thread in Java?
     → New, Runnable, Running, Blocked/Waiting, and Terminated — like a tiny worker with mood swings.

202. How can you check if a thread is alive or not?
     → Use `thread.isAlive()` to see if it’s still buzzing around.

203. What is the difference between `start()` and `run()` methods in threads?
     → `start()` creates a real new thread; `run()` is just a normal method pretending to be fancy.

204. Can you call `run()` directly instead of `start()`? What happens?
     → Yes, but it runs on the same thread, so no actual “multithreading magic” happens.

205. What is the life cycle of a thread?
     → It goes from born → runnable → running → waiting/blocking → done forever.

206. What happens when the `run()` method throws an exception?
     → The thread quietly dies with that exception as its last words.

207. How do you stop a thread gracefully?
     → Use a flag (like `isRunning`) and let the thread exit politely on its own.

208. What is the `volatile` keyword used for?
     → It makes a variable always read the fresh, up-to-date value from memory.

209. How is `volatile` different from `synchronized`?
     → `volatile` gives visibility, while `synchronized` gives both visibility and “one-at-a-time” access.

210. What is the difference between `Thread.sleep()` and `Object.wait()`?
     → `sleep()` just pauses; `wait()` pauses *and* gives up the lock like a polite guest.


### **Synchronization**

211. What is synchronization in Java?
     → It’s Java’s way of making threads take turns instead of wrestling over shared data.

212. Why is synchronization needed?
     → To stop threads from messing up shared variables like kids fighting over one toy.

213. What are intrinsic locks or monitor locks?
     → Built-in locks every Java object secretly carries around like a personal door key.

214. What is a synchronized method vs synchronized block?
     → A synchronized method locks the whole method; a block locks only the tiny part you choose.

215. Can static methods be synchronized?
     → Yes, they lock on the class itself like a big global gate.

216. What happens if one thread holds a lock and another tries to access it?
     → The second one just waits patiently like someone stuck outside a bathroom door.

217. What is a reentrant lock?
     → A lock that lets the same thread enter again without getting locked out by itself.

218. What are the disadvantages of synchronization?
     → Slower performance and the risk of deadlocks — like traffic jams for threads.

219. What is the difference between synchronized and concurrent collections?
     → Synchronized ones block everything; concurrent ones let more action happen safely at once.

220. What is thread interference?
     → When two threads scramble shared data by poking it at the same time.


### **Deadlock, Livelock & Starvation**

221. What is a deadlock?
     → It’s when two threads wait on each other forever like two people holding doors saying “you first.”

222. What are the four conditions that cause deadlock?
     → Mutual exclusion, hold-and-wait, no preemption, and circular waiting — the four horsemen of thread misery.

223. How do you detect a deadlock?
     → Look for threads stuck waiting on each other with no chance of escape.

224. How can you prevent a deadlock?
     → Break at least one of the four conditions by ordering locks or avoiding long lock-holds.

225. What is livelock?
     → Threads keep moving but never progress, like two people dancing sideways to avoid each other.

226. What is thread starvation?
     → When one thread never gets CPU time because others hog the spotlight.

227. What are some ways to avoid thread starvation?
     → Use fair locks, balanced priorities, and avoid greedy resource grabbing.

228. Can deadlocks occur with only one thread?
     → Nope, you need at least two troublemakers.

229. How do you debug deadlocks in Java?
     → Take thread dumps and look for threads stuck in `BLOCKED` with circular lock waits.

230. What tools can be used to identify deadlocks?
     → JDK tools like `jstack`, VisualVM, Mission Control, and IntelliJ/IDE debuggers.

### **Thread Communication**

231. What is inter-thread communication?
     → It’s how threads politely talk and coordinate instead of bumping into each other.

232. What are the three main methods of inter-thread communication?
     → `wait()`, `notify()`, and `notifyAll()` — the tiny messaging trio.

233. What is the difference between `notify()` and `notifyAll()`?
     → `notify()` wakes one waiter; `notifyAll()` wakes the whole sleepy crowd.

234. What happens if `notify()` is called with no waiting threads?
     → Nothing at all — it just vanishes into the void.

235. Why must `wait()` and `notify()` be called inside synchronized blocks?
     → Because they need the object’s lock to avoid wild, unsafe wake-ups.

236. What is spurious wakeup in Java?
     → When a thread wakes up even though nobody actually notified it — a random nudge.

237. How can you handle spurious wakeups?
     → Always wait in a loop that re-checks the condition.

238. Can `wait()` be interrupted?
     → Yes, and it throws `InterruptedException` to say so.

239. What happens if `wait()` is called on an object without holding its lock?
     → It throws `IllegalMonitorStateException` like a strict bouncer.

240. What is the purpose of the `join()` method in threads?
     → It makes one thread wait until another finishes its work.


### **Executor Framework**

241. What is the Executor Framework in Java?
     → A system that manages threads for you so you don’t juggle them manually.

242. Why was the Executor Framework introduced?
     → To simplify thread management and make concurrency less chaotic.

243. What is the difference between Executor and ExecutorService?
     → Executor just runs tasks; ExecutorService adds extras like shutdown and task tracking.

244. What is the role of `Callable` and `Future`?
     → `Callable` returns a value; `Future` holds that value for later pickup.

245. What is the difference between `Runnable` and `Callable`?
     → `Runnable` returns nothing; `Callable` returns something and can throw checked exceptions.

246. What is the `submit()` method used for?
     → To send a task to the executor and get a `Future` back.

247. What is a `FutureTask`?
     → A wrapper combining a task and its future result in one neat package.

248. How do you cancel a running task?
     → Call `future.cancel(true)` and hope the task behaves.

249. What are the different types of thread pools in `Executors`?
     → Fixed, cached, single-thread, and scheduled pools.

250. How does a `FixedThreadPool` work?
     → It uses a set number of threads that reuse themselves endlessly.

251. What is the difference between `CachedThreadPool` and `FixedThreadPool`?
     → Cached expands and shrinks freely; fixed stays the same size.

252. What is a `SingleThreadExecutor`?
     → A pool with exactly one worker that handles tasks one by one.

253. What is a `ScheduledThreadPoolExecutor`?
     → A pool that can delay or repeat tasks on a schedule.

254. How do you schedule a task for repeated execution?
     → Use `scheduleAtFixedRate()` or `scheduleWithFixedDelay()`.

255. What is the difference between `schedule()` and `scheduleAtFixedRate()`?
     → `schedule()` runs once; `scheduleAtFixedRate()` repeats at steady intervals.

256. What is the role of the `RejectedExecutionHandler`?
     → It decides what to do when the pool refuses a task.

257. How do you shut down an executor service?
     → Call `shutdown()` or `shutdownNow()` politely or forcefully.

258. What is the difference between `shutdown()` and `shutdownNow()`?
     → `shutdown()` lets tasks finish; `shutdownNow()` tries to stop everything instantly.

259. What happens if you submit a task after shutdown?
     → You get a rejection because the pool has closed shop.

260. How do you handle exceptions in threads using ExecutorService?
     → Catch them via `Future.get()` or wrap tasks with custom handlers.


### **Locks and Synchronizers**

261. What is the `Lock` interface in Java?
     → A manual lock you control yourself instead of relying on `synchronized`.

262. What is the difference between `Lock` and `synchronized`?
     → `Lock` gives more control; `synchronized` is automatic and simpler.

263. What is `ReentrantLock`?
     → A lock that lets the same thread re-enter without blocking itself.

264. What are the advantages of using `ReentrantLock`?
     → Try-locking, fair locking, interruptible locking, and more flexibility.

265. What is `tryLock()` used for?
     → To attempt a lock without waiting forever.

266. What is a fair lock?
     → A lock that hands out turns in strict arrival order.

267. What is a `ReadWriteLock`?
     → A lock with one write lock and many read locks.

268. When would you use a `ReadWriteLock`?
     → When reads are frequent and writes are rare.

269. What is the `StampedLock` in Java 8?
     → A lock with stamps that supports optimistic reads.

270. What is the difference between `StampedLock` and `ReentrantLock`?
     → `StampedLock` supports fast optimistic reads; `ReentrantLock` doesn’t.

271. What is `Condition` in Java concurrency?
     → A tool for finer control of waiting and signalling inside locks.

272. What is the difference between `Condition` and `Object.wait()`?
     → `Condition` works with Lock; `wait()` works with intrinsic locks.

273. How do you use a `Condition` to signal between threads?
     → Lock → wait with `await()` → signal with `signal()` or `signalAll()`.

274. What is a `Semaphore`?
     → A counter that limits how many threads can access something.

275. What is the difference between binary and counting semaphores?
     → Binary allows one permit; counting allows many.

276. What is a `CountDownLatch`?
     → A latch that releases threads once its count hits zero.

277. How is a `CountDownLatch` different from a `CyclicBarrier`?
     → Latch is one-time; barrier resets and can be reused.

278. What is a `CyclicBarrier` used for?
     → To make threads wait until all have reached the same point.

279. What is a `Phaser`?
     → A flexible barrier that supports dynamic registration of parties.

280. What is the difference between `Phaser` and `CyclicBarrier`?
     → Phaser is more flexible and reusable with changing participants.


### **Atomic and Concurrent Utilities**

281. What are atomic variables in Java?
     → Special variables that update safely without locks.

282. What is the difference between atomic and volatile variables?
     → Atomic changes values safely; volatile only guarantees visibility.

283. What is compare-and-swap (CAS)?
     → A fast trick to update a value only if it hasn’t changed.

284. What is the role of the `AtomicInteger` class?
     → To do thread-safe integer updates without locks.

285. What is the difference between `AtomicInteger` and `synchronized` increment?
     → Atomic uses CAS; synchronized uses locking.

286. What is the `ConcurrentLinkedQueue`?
     → A lock-free, thread-safe queue.

287. How is `ConcurrentHashMap` made thread-safe?
     → By using fine-grained locking and non-blocking reads.

288. What is the difference between `CopyOnWriteArrayList` and `Vector`?
     → CopyOnWrite copies on writes; Vector locks everything.

289. What is `ConcurrentSkipListMap`?
     → A sorted, thread-safe map based on skip-list structures.

290. When should you use `ConcurrentSkipListMap` instead of `ConcurrentHashMap`?
     → When you need sorted, navigable ordering.

291. What is `BlockingQueue` and its types?
     → A queue that waits on insert/remove; types include array, linked, priority, delay, and synchronous.

292. What is the difference between `ArrayBlockingQueue` and `LinkedBlockingQueue`?
     → Array is fixed-size; linked can grow larger.

293. What is the purpose of a `DelayQueue`?
     → To release items only after their delay expires.

294. What is the `SynchronousQueue`?
     → A queue with zero capacity that hands items directly from one thread to another.

295. What is the `Exchanger` class?
     → A tool for two threads to swap data.

296. What is the difference between `CountDownLatch` and `Exchanger`?
     → Latch waits for a countdown; Exchanger swaps data between two threads.

297. What is the difference between concurrent and parallel programming?
     → Concurrent means many tasks in progress; parallel means tasks run literally at the same time.

298. What are fork/join frameworks in Java?
     → A system for splitting big tasks into smaller ones and combining results.

299. What is the `RecursiveTask` in fork/join?
     → A task that returns a result after splitting and joining.

300. What is the difference between `ForkJoinPool` and `ExecutorService`?
     → ForkJoinPool uses work-stealing for divide-and-conquer tasks; ExecutorService uses simpler task execution.

---

## 💾 **Batch 4 — Java I/O, NIO, Serialization, and Networking (Q301–Q400)**

### **Java I/O (Streams and Readers/Writers)**

301. What is Java I/O?
     → It’s Java’s system for reading and writing data from sources like files, memory, and networks.

302. What are the main types of I/O streams in Java?
     → Byte streams and character streams.

303. What is the difference between byte streams and character streams?
     → Byte streams handle raw bytes; character streams handle text with proper encoding.

304. What are the main abstract classes for byte streams?
     → `InputStream` and `OutputStream`.

305. What are the main abstract classes for character streams?
     → `Reader` and `Writer`.

306. What is the difference between `InputStream` and `Reader`?
     → `InputStream` reads bytes; `Reader` reads characters.

307. What is the difference between `OutputStream` and `Writer`?
     → `OutputStream` writes bytes; `Writer` writes characters.

308. What is the role of `FileInputStream` and `FileOutputStream`?
     → They read and write raw byte data to files.

309. How does `BufferedInputStream` improve performance?
     → It reduces disk reads by loading chunks of data into a buffer.

310. What is the difference between `BufferedReader` and `FileReader`?
     → `BufferedReader` adds efficient line-by-line reading on top of a reader like `FileReader`.

311. What is the purpose of `PrintWriter`?
     → It provides easy text printing with convenience methods.

312. What is the difference between `flush()` and `close()`?
     → `flush()` pushes pending data out; `close()` flushes and then terminates the stream.

313. What happens if you write to a stream after closing it?
     → It throws an `IOException`.

314. What is the purpose of `DataInputStream` and `DataOutputStream`?
     → They read and write Java primitive data types in a portable form.

315. What are filter streams in Java?
     → Streams that wrap other streams to add extra features like buffering or type handling.

316. What is `SequenceInputStream` used for?
     → To read multiple input streams as if they were one continuous stream.

317. What is the difference between `pushback` and `buffered` streams?
     → Pushback lets you “unread” bytes; buffered improves speed by storing chunks.

318. What is `ObjectInputStream` used for?
     → For reading serialized Java objects back into memory.

319. What is the purpose of `InputStreamReader` and `OutputStreamWriter`?
     → They convert between byte streams and character streams using a chosen encoding.

320. How does character encoding affect file reading and writing?
     → It determines how characters map to bytes, influencing correctness of text data.


### **File Handling**

321. How do you create a file in Java?
     → Use `new File("name").createNewFile()`.

322. How do you delete a file in Java?
     → Call `file.delete()`.

323. How do you check if a file exists?
     → Use `file.exists()`.

324. What is the `File` class in Java used for?
     → To represent file and directory paths.

325. How do you list all files in a directory?
     → Use `file.list()` or `file.listFiles()`.

326. How do you create a directory in Java?
     → Call `file.mkdir()` or `file.mkdirs()`.

327. Can you rename a file using Java?
     → Yes, with `file.renameTo(newFile)`.

328. What is the difference between absolute and relative file paths?
     → Absolute gives full location; relative depends on current working directory.

329. What happens if you try to read from a non-existent file?
     → You get a `FileNotFoundException`.

330. What are some common exceptions in file handling?
     → `IOException`, `FileNotFoundException`, `SecurityException`.

331. How do you handle large files efficiently?
     → Use buffered streams or NIO channels.

332. What are the advantages of using buffered streams for file operations?
     → Faster reads/writes by reducing direct disk access.

333. How do you copy a file using Java I/O?
     → Read from input stream and write to output stream in a loop.

334. How do you move a file in Java?
     → Use `Files.move()` in NIO.

335. How do you append data to a file?
     → Use `FileWriter(file, true)` or `Files.write(..., APPEND)`.

336. What is the difference between `FileWriter` and `FileOutputStream`?
     → `FileWriter` writes characters; `FileOutputStream` writes bytes.

337. What are the limitations of the `java.io.File` API?
     → Limited metadata handling and no real file operations like copy/move.

338. How do you check if a file is readable or writable?
     → Use `file.canRead()` and `file.canWrite()`.

339. How do you set file permissions in Java?
     → Use `file.setReadable()`, `setWritable()`, or NIO `PosixFilePermissions`.

340. What is `Files` class in Java NIO used for?
     → High-level operations like copy, move, delete, read, and write.


### **Java NIO (New I/O)**

341. What is Java NIO?
     → It’s a faster, scalable I/O system using buffers, channels, and selectors.

342. Why was NIO introduced when we already had I/O?
     → To support non-blocking, high-performance, multiplexed I/O.

343. What are buffers in NIO?
     → Memory containers that hold data read from or written to channels.

344. What are channels in NIO?
     → Bidirectional data pipes that connect buffers to I/O sources.

345. What is the difference between streams and channels?
     → Streams are one-way; channels are two-way and buffer-based.

346. What is the role of the `ByteBuffer` class?
     → It stores bytes and provides methods to read/write them efficiently.

347. What is a direct buffer?
     → A buffer allocated outside the JVM heap for faster native I/O.

348. What are the types of buffers available in NIO?
     → Byte, Char, Short, Int, Long, Float, and Double buffers.

349. How do you read data from a channel into a buffer?
     → Call `channel.read(buffer)`.

350. What are non-blocking channels?
     → Channels that return immediately without waiting for data readiness.

351. What is the purpose of `FileChannel`?
     → To perform high-speed file operations using buffers.

352. How do you transfer data between channels?
     → Using `transferTo()` or `transferFrom()`.

353. What is the difference between scatter and gather operations?
     → Scatter reads into multiple buffers; gather writes from multiple buffers.

354. What is a selector in NIO?
     → A component that monitors multiple channels for I/O readiness.

355. What is the role of the `SelectionKey` class?
     → It represents a channel’s registration and readiness state with a selector.

356. What are the possible selection operations in NIO?
     → `OP_READ`, `OP_WRITE`, `OP_ACCEPT`, and `OP_CONNECT`.

357. What is the difference between blocking and non-blocking I/O?
     → Blocking waits for data; non-blocking returns instantly.

358. How do selectors help in multiplexed I/O?
     → They let one thread manage many channels efficiently.

359. What is asynchronous I/O in Java NIO.2?
     → I/O operations that run independently and signal completion later.

360. What is the `Path` interface used for?
     → Representing file system paths in a modern, flexible way.


### **NIO.2 (File and Path API)**

361. What is the `Paths` class used for?
     → To create `Path` objects easily.

362. How do you create a path using NIO.2?
     → Use `Paths.get("path")`.

363. What is the `Files.walk()` method used for?
     → To recursively traverse directories.

364. What is the difference between `Files.readAllBytes()` and `Files.lines()`?
     → One returns raw bytes; the other returns a stream of text lines.

365. What is the `FileVisitor` interface used for?
     → Custom directory walking logic.

366. What is `SimpleFileVisitor`?
     → A helper class with default `FileVisitor` implementations.

367. How do you read file attributes in NIO.2?
     → Use `Files.readAttributes(path, BasicFileAttributes.class)`.

368. What is the `BasicFileAttributes` interface?
     → A view of common metadata like size, timestamps, and type.

369. How do you copy files using `Files.copy()`?
     → Call `Files.copy(src, dest, options...)`.

370. What is symbolic linking in NIO.2?
     → Creating a shortcut-like path pointing to another file.

371. How do you watch directory changes in NIO.2?
     → By using a `WatchService`.

372. What is the `WatchService` API?
     → A system for monitoring file events in directories.

373. What are the possible events in `WatchService`?
     → `ENTRY_CREATE`, `ENTRY_MODIFY`, and `ENTRY_DELETE`.

374. How do you register a directory with `WatchService`?
     → Use `path.register(watchService, events...)`.

375. What happens when a watched file is modified or deleted?
     → The watch key receives the corresponding event.

376. What are the advantages of NIO.2 over traditional I/O?
     → Better file APIs, async I/O, and robust path handling.

377. What is the difference between asynchronous and synchronous channels?
     → Sync waits for completion; async continues immediately.

378. What is the `AsynchronousFileChannel`?
     → A channel for non-blocking file operations.

379. How do you perform asynchronous read/write in NIO.2?
     → Call `read()` or `write()` with a `CompletionHandler`.

380. What is the use of `CompletionHandler`?
     → To receive callbacks when async operations finish.


### **Serialization**

381. What is serialization in Java?
     → Converting an object into a byte stream.

382. Why do we need serialization?
     → To store or send objects across systems.

383. How do you make a class serializable?
     → Implement `Serializable`.

384. What is the role of the `Serializable` interface?
     → It marks a class as eligible for serialization.

385. What happens if a superclass is not serializable?
     → Its fields are skipped and must be reinitialized.

386. What is the `serialVersionUID`?
     → A version marker for serialized classes.

387. What happens if `serialVersionUID` is not declared?
     → Java generates one, risking mismatch errors.

388. How do you prevent serialization of certain fields?
     → Mark them `transient`.

389. What is the `transient` keyword used for?
     → To exclude a field from serialization.

390. What is the difference between `Serializable` and `Externalizable`?
     → `Externalizable` gives full manual control.

391. How do you customize serialization?
     → Implement `writeObject()` and `readObject()`.

392. What are some common issues with serialization?
     → Version conflicts, performance cost, security risks.

393. What is deep serialization?
     → Serializing an object and all referenced objects.

394. Can static fields be serialized?
     → No, because they belong to the class.

395. How do you serialize a list of objects?
     → Write the list itself with `ObjectOutputStream`.

396. What is object deserialization?
     → Rebuilding an object from its byte stream.

397. What is the risk of deserializing untrusted data?
     → Possible remote code execution.

398. What are alternatives to Java’s built-in serialization?
     → JSON, XML, Protobuf, Kryo, Avro.

399. How do you compress serialized data?
     → Wrap streams with `GZIPOutputStream`.

400. What libraries are commonly used for faster or safer serialization?
     → Kryo, Protobuf, Avro, Jackson.


---

## 🏦 **Batch 5 — JDBC, ORM, and Transaction Management (Q401–Q500)**

### **JDBC Basics**

401. What is JDBC?
     → It’s Java’s way of talking to databases.

402. What are the main steps to connect a Java application to a database using JDBC?
     → Load driver, get connection, create statement, run query, close everything.

403. What are the core components of JDBC?
     → DriverManager, Connection, Statement, ResultSet.

404. What is the role of the `DriverManager` class?
     → It hands out database connections like a polite receptionist.

405. What are JDBC drivers?
     → Little connectors that let Java talk to specific databases.

406. What are the different types of JDBC drivers?
     → Type 1, Type 2, Type 3, Type 4.

407. What is the difference between Type 1, Type 2, Type 3, and Type 4 JDBC drivers?
     → Type 1 uses ODBC, Type 2 uses native APIs, Type 3 uses middleware, Type 4 talks directly to the DB.

408. Which type of JDBC driver is platform-independent?
     → Type 4.

409. What is the `Connection` interface in JDBC used for?
     → To open a communication line with the database.

410. How do you establish a JDBC connection to a database?
     → Use `DriverManager.getConnection(url, user, pass)`.

411. What is a connection string (JDBC URL)?
     → A special address telling Java where and how to reach your database.

412. What are some common JDBC connection URLs?
     → `jdbc:mysql://…`, `jdbc:postgresql://…`, `jdbc:oracle:thin:@…`.

413. What is the `Statement` interface used for?
     → Running SQL commands.

414. What is the difference between `Statement`, `PreparedStatement`, and `CallableStatement`?
     → Statement is basic, PreparedStatement is precompiled, CallableStatement calls stored procedures.

415. What is SQL injection, and how does `PreparedStatement` prevent it?
     → A sneaky input attack; PreparedStatement separates data from SQL code.

416. What is the advantage of using `PreparedStatement` over `Statement`?
     → Faster and safer.

417. How do you execute a query in JDBC?
     → Call `executeQuery()` or `executeUpdate()`.

418. What is the `ResultSet` interface?
     → A table-like view of your query results.

419. How do you iterate over a `ResultSet`?
     → Use a `while (rs.next())` loop.

420. What are the different types of `ResultSet`?
     → Forward-only, scrollable, and updatable.


### **Advanced JDBC Concepts**

421. What is the difference between `execute()`, `executeQuery()`, and `executeUpdate()`?
     → `execute()` handles any SQL, `executeQuery()` returns results, `executeUpdate()` returns affected rows.

422. What are batch updates in JDBC?
     → Grouped SQL operations sent together.

423. How do you execute batch updates?
     → Add statements with `addBatch()` and run them with `executeBatch()`.

424. What is connection pooling?
     → Reusing a set of open database connections.

425. What is the benefit of connection pooling?
     → Faster performance and lower resource cost.

426. What is `DataSource` in JDBC?
     → An object that hands out connections in a managed way.

427. How does `DataSource` differ from `DriverManager`?
     → It supports pooling and is container-friendly.

428. What is JNDI and how does it relate to `DataSource`?
     → A naming service used to look up `DataSource` objects.

429. What is auto-commit mode in JDBC?
     → Each SQL statement is committed automatically.

430. How do you disable auto-commit mode?
     → Call `conn.setAutoCommit(false)`.

431. How do you perform manual transaction management in JDBC?
     → Disable auto-commit, run SQL, then call `commit()` or `rollback()`.

432. What are `commit()` and `rollback()` used for?
     → Commit saves changes; rollback undoes them.

433. What happens if a `commit()` is not called?
     → Changes are lost when the connection closes.

434. What is the difference between savepoint and rollback?
     → A savepoint marks a spot; rollback returns to it.

435. How do you create a savepoint in JDBC?
     → Use `conn.setSavepoint()`.

436. What is metadata in JDBC?
     → Information about the database and results.

437. What are `DatabaseMetaData` and `ResultSetMetaData`?
     → One describes the DB; the other describes result columns.

438. How can you retrieve column names dynamically from a query?
     → Use `ResultSetMetaData.getColumnName()`.

439. What are scrollable and updatable `ResultSet`s?
     → Scrollable moves freely; updatable allows edits.

440. What is the difference between forward-only and scroll-sensitive `ResultSet`s?
     → Forward-only moves one way; scroll-sensitive reacts to underlying data changes.


### **JDBC Best Practices & Performance**

441. What are some common JDBC performance issues?
     → Slow queries, too many connections, unclosed resources.

442. How do you handle resource cleanup in JDBC?
     → Close ResultSet, Statement, and Connection properly.

443. What is the try-with-resources statement and how does it help?
     → It auto-closes JDBC objects for you.

444. How do you log SQL queries executed through JDBC?
     → Use logging frameworks or driver-level logging.

445. What is connection leak detection?
     → Spotting connections that were opened but never closed.

446. What are common causes of connection leaks?
     → Missing `close()`, exceptions, or bad code paths.

447. How can you tune JDBC performance?
     → Use pooling, batching, indexing, and prepared statements.

448. What is the difference between JDBC batch update and bulk insert?
     → Batch sends many statements; bulk insert is DB-specific high-speed insertion.

449. What is the impact of using large transactions in JDBC?
     → More locks, more memory, slower performance.

450. How can you improve database connection performance?
     → Use pooling, reduce round-trips, and optimize queries.


### **ORM Concepts (Object-Relational Mapping)**

451. What is ORM (Object-Relational Mapping)?
     → A way to map Java objects to database tables automatically.

452. What problem does ORM solve?
     → The mismatch between object models and relational databases.

453. What are some popular ORM frameworks in Java?
     → Hibernate, JPA, EclipseLink, Spring Data JPA.

454. What is the difference between JDBC and ORM?
     → JDBC is manual SQL; ORM handles SQL for you.

455. What is the role of JPA in ORM?
     → It defines the standard rules that ORM tools follow.

456. What is the Java Persistence API (JPA)?
     → A specification for managing Java object persistence.

457. What is the difference between JPA and Hibernate?
     → JPA is a standard; Hibernate is its implementation.

458. What are entities in JPA?
     → Java classes mapped to database tables.

459. What is the `@Entity` annotation used for?
     → Marking a class as a persistent table-backed entity.

460. What is the role of the `@Id` annotation?
     → It marks the primary key field.

461. What is the purpose of the `@GeneratedValue` annotation?
     → Auto-generating primary key values.

462. What are entity relationships in JPA?
     → Links between entities that reflect table relationships.

463. What are the types of entity relationships?
     → One-to-One, One-to-Many, Many-to-One, Many-to-Many.

464. What is the difference between `@OneToOne` and `@OneToMany`?
     → One-to-one links single entities; one-to-many links a parent to many children.

465. What is a bidirectional relationship in JPA?
     → A relationship navigable from both sides.

466. What is the `mappedBy` attribute used for?
     → To mark the owning side of a bidirectional relationship.

467. What is the difference between `fetch = FetchType.LAZY` and `fetch = FetchType.EAGER`?
     → LAZY loads later; EAGER loads immediately.

468. What is cascading in JPA?
     → Automatic propagation of operations to related entities.

469. What are the different cascade types available in JPA?
     → PERSIST, MERGE, REMOVE, REFRESH, DETACH, ALL.

470. What is orphan removal in JPA?
     → Auto-deleting child entities no longer referenced.


### **Hibernate Framework**

471. What is Hibernate?
     → A powerful ORM framework that maps Java objects to database tables.

472. What are the advantages of Hibernate over JDBC?
     → Less SQL, cleaner code, caching, and automatic table-object mapping.

473. What are the main components of the Hibernate architecture?
     → SessionFactory, Session, Transaction, Query, and mapping files.

474. What is the role of `SessionFactory`?
     → It creates Sessions and holds heavy configuration.

475. What is the `Session` interface used for?
     → Performing CRUD and database operations.

476. What is the difference between `Session` and `EntityManager`?
     → Session is Hibernate-specific; EntityManager is JPA-standard.

477. How do you persist an entity using Hibernate?
     → Call `session.save()` or `persist()` inside a transaction.

478. What are Hibernate mappings?
     → Rules that link classes to tables and fields to columns.

479. What are HQL and Criteria API?
     → Hibernate’s query language and its programmatic query builder.

480. What is the difference between HQL and SQL?
     → HQL uses entity names; SQL uses table names.

481. What are named queries in Hibernate?
     → Predefined reusable HQL or SQL queries.

482. What is the difference between `merge()` and `update()` in Hibernate?
     → `merge()` copies changes safely; `update()` reattaches directly.

483. What is the difference between `save()`, `persist()`, and `saveOrUpdate()`?
     → `save()` returns ID, `persist()` follows JPA rules, `saveOrUpdate()` chooses insert or update.

484. What are transient, persistent, and detached states in Hibernate?
     → New object, managed object, and previously managed but now disconnected.

485. What is lazy loading in Hibernate?
     → Loading data only when it’s actually accessed.

486. What is the N+1 select problem?
     → Too many repeated queries for related entities.

487. How can you solve the N+1 select problem in Hibernate?
     → Use JOIN FETCH, batching, or entity graphs.

488. What is caching in Hibernate?
     → Storing data to reduce database trips.

489. What are the types of caching in Hibernate?
     → First-level, second-level, and query cache.

490. What is the difference between first-level and second-level cache?
     → First-level is per-session; second-level is shared across sessions.

491. What is the purpose of the query cache?
     → Caching results of frequently run queries.

492. What cache providers does Hibernate support?
     → Ehcache, Infinispan, Hazelcast, Redis-based providers.

493. What is dirty checking in Hibernate?
     → Auto-detection of changed fields for update.

494. How does Hibernate manage transactions?
     → Through `Transaction` objects tied to the Session.

495. What is optimistic locking?
     → Conflict prevention using version checks.

496. What is pessimistic locking?
     → Locking rows directly to avoid conflicts.

497. What is the difference between the two locking mechanisms?
     → Optimistic avoids locks; pessimistic uses them.

498. What is Hibernate Validator?
     → Hibernate’s framework for bean validation.

499. What are some common annotations used for validation?
     → `@NotNull`, `@Size`, `@Email`, `@Min`, `@Max`.

500. What are the advantages and disadvantages of using Hibernate?
     → Pros: less SQL, caching, portability; cons: overhead and learning curve.


---

## 🌱 **Batch 6 of 10 — Spring Core, IoC, DI, and AOP (Q501–Q600)**

### **Spring Framework Basics**

501. What is the Spring Framework?
     → A big helper toolkit that makes building Java apps easier and cleaner.

502. Why is Spring so popular in enterprise development?
     → It simplifies complex tasks and keeps apps flexible and modular.

503. What are the key features of the Spring Framework?
     → IoC, DI, AOP, data access, web support, and tons of utilities.

504. What are the different modules of Spring?
     → Core, AOP, Data, MVC, Security, Boot, and more.

505. What is Inversion of Control (IoC)?
     → Letting Spring create and manage objects instead of you.

506. How does IoC differ from traditional programming approaches?
     → You don’t “new” things; Spring hands them to you.

507. What is Dependency Injection (DI)?
     → Spring plugging needed objects into your classes automatically.

508. How does Dependency Injection improve testability and maintainability?
     → It avoids tight wiring, making code easier to swap and test.

509. What are the types of Dependency Injection supported in Spring?
     → Constructor, setter, and field injection.

510. What is the difference between constructor injection and setter injection?
     → Constructor sets everything upfront; setter sets things later.

511. What are the benefits of constructor-based DI?
     → Strong immutability and guaranteed dependencies.

512. What are the benefits of setter-based DI?
     → More flexibility and optional dependencies.

513. What is the difference between IoC and DI?
     → IoC is the big idea; DI is one way to do it.

514. What is a Spring Bean?
     → Any object Spring creates and manages for you.

515. How does Spring manage beans internally?
     → Through its container, which builds, wires, and tracks them.

516. What is the role of the ApplicationContext in Spring?
     → It’s the boss that loads beans and handles config.

517. How does ApplicationContext differ from BeanFactory?
     → ApplicationContext has more features like events and AOP.

518. What are some common implementations of ApplicationContext?
     → `ClassPathXmlApplicationContext`, `AnnotationConfigApplicationContext`, `WebApplicationContext`.

519. What is the difference between `AnnotationConfigApplicationContext` and `ClassPathXmlApplicationContext`?
     → One uses annotations; the other uses XML.

520. What are bean definitions in Spring?
     → Instructions telling Spring how to build a bean.


### **Bean Configuration & Lifecycle**

521. What are the different ways to configure Spring beans?
     → XML, Java classes, and annotations.

522. What is XML-based configuration in Spring?
     → Defining beans and wiring inside XML files.

523. What is Java-based configuration in Spring?
     → Using `@Configuration` classes and `@Bean` methods.

524. What is annotation-based configuration?
     → Marking classes with annotations so Spring auto-detects them.

525. What are the advantages of annotation-based configuration?
     → Cleaner code and fewer external files.

526. What is the `@Configuration` annotation used for?
     → Marking a class as a bean-creating config source.

527. What is the `@Bean` annotation used for?
     → Telling Spring to create and manage a specific object.

528. What is the `@Component` annotation?
     → A general marker for classes Spring should manage.

529. What is the difference between `@Component`, `@Service`, `@Repository`, and `@Controller`?
     → Same idea, different roles: generic, business, DB, and web.

530. What is component scanning?
     → Spring searching packages to find annotated beans.

531. What is the `@ComponentScan` annotation used for?
     → Telling Spring where to look for components.

532. What are the different bean scopes in Spring?
     → Singleton, prototype, request, session, and application.

533. What is the difference between `singleton` and `prototype` scope?
     → Singleton gives one instance; prototype gives a new one each time.

534. What are `request`, `session`, and `application` scopes used for?
     → Web apps: per-request, per-session, and per-servlet-context beans.

535. How does Spring handle bean lifecycle?
     → It creates, wires, initializes, uses, and destroys beans.

536. What are the callback methods in Spring bean lifecycle?
     → Init methods, destroy methods, and post-processors.

537. What is the `InitializingBean` interface used for?
     → For custom initialization code.

538. What is the `DisposableBean` interface used for?
     → For custom cleanup code.

539. What is the `@PostConstruct` annotation used for?
     → Running code right after bean creation.

540. What is the `@PreDestroy` annotation used for?
     → Running code before the bean is removed.


### **Dependency Injection Details**

541. What is autowiring in Spring?
     → Spring automatically injects required beans without manual wiring.

542. What are the types of autowiring modes in Spring?
     → By type, by name, constructor, and autodetect (older XML modes).

543. What is the `@Autowired` annotation?
     → A marker telling Spring to inject a matching bean.

544. How does Spring resolve dependencies with multiple beans of the same type?
     → By using qualifiers, primary beans, or bean names.

545. What is the `@Qualifier` annotation used for?
     → To pick the exact bean you want when several match.

546. What is the `@Primary` annotation used for?
     → To mark one bean as Spring’s default choice.

547. What is the difference between field injection and constructor injection using annotations?
     → Field injects directly; constructor injects through parameters.

548. Why is constructor injection preferred over field injection?
     → It ensures required dependencies and makes testing easier.

549. What happens if Spring cannot autowire a dependency?
     → It throws a `NoSuchBeanDefinitionException`.

550. What is circular dependency, and how does Spring handle it?
     → Two beans depend on each other; Spring resolves some cases via proxies or fails if impossible.


### **Spring Expression Language (SpEL)**

551. What is the Spring Expression Language (SpEL)?
     → A mini-language for dynamic values in Spring.

552. How is SpEL used in Spring applications?
     → Inside annotations, XML, or config to compute values.

553. What are some common use cases for SpEL?
     → Reading properties, calling methods, doing math, conditional logic.

554. How do you enable SpEL in configuration?
     → Use `#{...}` inside bean definitions.

555. What are the operators supported by SpEL?
     → Math, logical, relational, and conditional operators.

556. Can SpEL access properties or method results?
     → Yes, it can read fields and call methods.

557. Can SpEL be used in annotations?
     → Yes, especially with `@Value`.

558. What is the difference between `${}` and `#{}` in Spring configuration?
     → `${}` reads properties; `#{}` evaluates SpEL expressions.

559. How do you inject values from a properties file using SpEL?
     → Combine `@Value("${key}")` with property sources.

560. What are environment properties in Spring?
     → External config values loaded from the system, files, or profiles.


### **Spring AOP (Aspect-Oriented Programming)**

561. What is AOP (Aspect-Oriented Programming)?
     → A way to add common behaviors across methods without cluttering code.

562. Why is AOP needed in Spring?
     → To cleanly handle things like logging, security, and transactions.

563. What are cross-cutting concerns?
     → Features that affect many parts of an app.

564. What are the key concepts in AOP?
     → Aspects, join points, pointcuts, advices, and weaving.

565. What is an Aspect?
     → A module that bundles cross-cutting logic.

566. What is a Join Point?
     → A place in code where extra behavior can run.

567. What is a Pointcut?
     → A rule that selects which join points to target.

568. What is an Advice?
     → The actual code that runs at matched join points.

569. What are the different types of Advices in AOP?
     → Before, After, AfterReturning, AfterThrowing, Around.

570. What is a Weaving process in AOP?
     → The act of applying aspects to code.

571. What are the types of weaving?
     → Compile-time, load-time, and runtime.

572. What is the difference between compile-time, load-time, and runtime weaving?
     → When the aspect gets applied: during compile, class loading, or execution.

573. What is the `@Aspect` annotation used for?
     → Marking a class as an aspect container.

574. What is the `@Before` annotation used for?
     → Running logic before a method executes.

575. What is the `@After` annotation used for?
     → Running logic after a method finishes (success or failure).

576. What is the `@AfterReturning` annotation?
     → Running logic only when a method returns normally.

577. What is the `@AfterThrowing` annotation?
     → Running logic when a method throws an exception.

578. What is the `@Around` advice?
     → Code that wraps around a method call.

579. How does `@Around` advice differ from other advices?
     → It can block, modify, or proceed with execution.

580. What is the role of `ProceedingJoinPoint`?
     → It lets `@Around` advice control the actual method call.


### **Advanced Spring AOP & Proxy Concepts**

581. What is proxy-based AOP in Spring?
     → Spring creates proxy objects that wrap your beans to apply AOP logic.

582. What is the difference between JDK dynamic proxy and CGLIB proxy?
     → JDK proxies work with interfaces; CGLIB proxies subclass concrete classes.

583. When does Spring use JDK proxy vs CGLIB proxy?
     → JDK if you use interfaces; CGLIB if no interfaces exist.

584. What are the limitations of Spring AOP compared to AspectJ?
     → Spring can advise only methods, not constructors or fields.

585. What is AspectJ, and how does it differ from Spring AOP?
     → AspectJ is a full AOP framework with deeper weaving power.

586. What is the difference between declarative and programmatic AOP?
     → Declarative uses annotations; programmatic calls advice manually.

587. Can we apply multiple advices to a single method?
     → Yes, and Spring runs them in order.

588. How does Spring order multiple aspects?
     → By priority rules or annotations.

589. What is the purpose of the `@Order` annotation in AOP?
     → It sets which aspect runs first.

590. What is a custom annotation-based aspect?
     → An aspect triggered by your own custom-made annotation.


### **Spring Core Utilities & Context**

591. What is the Environment abstraction in Spring?
     → A helper that gives access to properties, profiles, and environment details.

592. What is a PropertySource in Spring?
     → A place where Spring loads key–value configuration from.

593. How do you externalize configuration in Spring?
     → By moving settings to properties/YAML files or environment variables.

594. What is the role of `@PropertySource`?
     → To load a specific properties file into the Spring Environment.

595. What is the `MessageSource` used for?
     → Handling text messages, especially for localization.

596. What is internationalization (i18n) in Spring?
     → Making apps support multiple languages and cultures.

597. How do you support i18n in Spring applications?
     → Use `MessageSource` and locale resolvers.

598. What is an ApplicationEvent in Spring?
     → A message that signals something happened inside the app.

599. What is the difference between standard Java events and Spring events?
     → Spring events are simpler, loosely coupled, and container-managed.

600. How do you publish and listen to custom events in Spring?
     → Publish with `ApplicationEventPublisher` and handle with `@EventListener`.


---

## 🚀 **Batch 7 — Spring Boot, REST, and Configuration (Q601–Q700)**

### **Spring Boot Basics**

601. What is Spring Boot?
     → A tool that helps you build Spring apps quickly with minimal setup.

602. How is Spring Boot different from the Spring Framework?
     → Boot auto-configures things; Spring needs more manual setup.

603. What are the main advantages of using Spring Boot?
     → Faster development, fewer configs, built-in servers, easy packaging.

604. What is auto-configuration in Spring Boot?
     → Boot guessing and setting things up for you automatically.

605. What is the role of `@SpringBootApplication`?
     → It marks the main class and turns on key Boot features.

606. What three annotations are combined in `@SpringBootApplication`?
     → `@Configuration`, `@EnableAutoConfiguration`, `@ComponentScan`.

607. What is a Spring Boot starter?
     → A ready-made dependency bundle for common features.

608. Name some commonly used Spring Boot starters.
     → Web, Data JPA, Security, Test, Thymeleaf.

609. How does Spring Boot simplify dependency management?
     → By using curated starter packs with fixed versions.

610. What is the role of `application.properties` or `application.yml`?
     → They hold your app’s external configuration.

611. How do you run a Spring Boot application?
     → Use the main method, Maven/Gradle, or jar execution.

612. What is an embedded server in Spring Boot?
     → A built-in web server inside your app.

613. Which embedded servers are supported by Spring Boot?
     → Tomcat, Jetty, and Undertow.

614. What is the default embedded server in Spring Boot?
     → Tomcat.

615. Can Spring Boot run without an embedded server?
     → Yes, if it’s not a web app.

616. How do you change the server port in Spring Boot?
     → Set `server.port` in properties or YAML.

617. What is the difference between Spring Boot CLI and Spring Boot Starter?
     → CLI runs apps from the command line; starters manage dependencies.

618. What are the profiles in Spring Boot?
     → Named sets of configuration for different environments.

619. How do you activate a profile in Spring Boot?
     → Set `spring.profiles.active` in config or CLI.

620. What is the role of the `@Profile` annotation?
     → It enables a bean only for certain profiles.


### **Spring Boot Configuration & Properties**

621. What is externalized configuration in Spring Boot?
     → Storing settings outside the code so they’re easy to change.

622. What is the difference between `application.properties` and `application.yml`?
     → Same purpose, different formats—properties is key=value, YAML is structured.

623. How do you load properties from a custom file?
     → Use `@PropertySource` or put it in the config path.

624. What is the `@Value` annotation used for?
     → Injecting simple values from properties.

625. How do you bind a configuration to a POJO?
     → Use `@ConfigurationProperties`.

626. What is `@ConfigurationProperties`?
     → A way to map groups of configs to a class.

627. How do you validate configuration properties?
     → Add validation annotations and enable validation.

628. What is the difference between `@PropertySource` and `@ConfigurationProperties`?
     → One loads files; the other binds values into objects.

629. How do you override properties using command-line arguments?
     → Pass them as `--key=value` when starting the app.

630. What is the difference between profile-specific properties and default properties?
     → Default applies to all; profile-specific applies only when that profile is active.


### **Spring Boot Actuator & Monitoring**

631. What is Spring Boot Actuator?
     → A toolkit that gives you live info and monitoring for your app.

632. How do you enable Actuator endpoints?
     → Add the Actuator starter and configure `management.endpoints` in properties.

633. Name some common Actuator endpoints.
     → `/health`, `/info`, `/metrics`, `/env`, `/beans`.

634. How do you secure Actuator endpoints?
     → Use Spring Security and restrict access rules.

635. What is health check in Actuator?
     → A quick report telling if your app is running fine.

636. How can you customize health indicators?
     → Create your own `HealthIndicator` beans.

637. What is the difference between public and sensitive endpoints?
     → Public is harmless; sensitive exposes deeper system details.

638. What is the `/info` endpoint used for?
     → Showing app metadata like version or description.

639. What is the `/metrics` endpoint used for?
     → Displaying numbers about app performance and resources.

640. How do you expose custom metrics using Actuator?
     → Register counters or gauges with Micrometer.


### **Spring Boot REST**

641. What is REST?
     → A style for building web APIs using simple HTTP rules.

642. What are the principles of REST architecture?
     → Statelessness, resources, URIs, standard verbs, and uniform responses.

643. How do you create a REST controller in Spring Boot?
     → Use `@RestController` and mapping annotations.

644. What is the difference between `@Controller` and `@RestController`?
     → RestController auto-returns JSON; Controller needs `@ResponseBody`.

645. How do you map HTTP methods to controller methods?
     → With `@GetMapping`, `@PostMapping`, and friends.

646. What is the difference between `@RequestMapping` and the specialized annotations like `@GetMapping`?
     → RequestMapping is generic; specialized ones are shortcuts.

647. How do you pass path variables in Spring Boot REST?
     → Put them in the URL and grab them with `@PathVariable`.

648. What is the `@PathVariable` annotation used for?
     → Capturing values from the URL path.

649. How do you pass query parameters?
     → Add them to the URL and read with `@RequestParam`.

650. What is the `@RequestParam` annotation used for?
     → Extracting query parameters.

651. How do you handle POST requests with JSON payload?
     → Use `@PostMapping` and bind data with `@RequestBody`.

652. What is the `@RequestBody` annotation used for?
     → Turning JSON into a Java object.

653. How do you return JSON responses?
     → Return objects from `@RestController` methods.

654. What is the `@ResponseBody` annotation?
     → It tells Spring to write the return value as JSON/XML.

655. How do you handle custom response statuses?
     → Use `@ResponseStatus` or return `ResponseEntity`.

656. What is the `@ResponseStatus` annotation?
     → A way to set a fixed HTTP status for a method.

657. How do you handle exceptions in REST controllers?
     → Create handlers using `@ExceptionHandler`.

658. What is `@ControllerAdvice` used for?
     → Centralized, global exception handling.

659. What is `@ExceptionHandler`?
     → A method that handles a specific exception type.

660. How do you validate REST request payloads?
     → Use `@Valid` with validation annotations.


### **Spring Boot Data & Repositories**

661. What is Spring Data JPA?
     → A helper library that makes JPA database work super easy.

662. How do you integrate Spring Data JPA with Spring Boot?
     → Add the JPA starter and set DB properties.

663. What is a Spring Data repository?
     → An interface that gives you ready-made CRUD operations.

664. What is the difference between `CrudRepository`, `JpaRepository`, and `PagingAndSortingRepository`?
     → Crud = basic ops, PagingAndSorting = adds paging, JpaRepository = adds JPA extras.

665. How do you define custom queries in Spring Data JPA?
     → Use query methods or the `@Query` annotation.

666. What is the `@Query` annotation used for?
     → Writing your own JPQL or SQL queries.

667. How do you use derived query methods in Spring Data?
     → By naming methods like `findByName` or `findByAgeGreaterThan`.

668. How do you implement pagination in Spring Data?
     → Pass a `PageRequest` to repository methods.

669. How do you implement sorting in Spring Data?
     → Use a `Sort` object in your calls.

670. What is the difference between `save()` and `saveAndFlush()`?
     → `save()` waits to write; `saveAndFlush()` writes immediately.


### **Spring Boot Security Basics**

671. How do you enable Spring Security in a Spring Boot application?
     → Just add the Spring Security starter and boom—security turns itself on.

672. What is the default security configuration in Spring Boot?
     → Everything is locked down with a generated password and basic auth.

673. How do you customize authentication in Spring Boot?
     → Provide your own security config class and override the rules.

674. How do you configure in-memory users?
     → Define them in a security config using an in-memory user details manager.

675. How do you configure JDBC-based authentication?
     → Point Spring Security to a database and let it read users and roles.

676. What is the role of `PasswordEncoder`?
     → It scrambles passwords so they aren’t stored like plain text.

677. How do you secure REST endpoints?
     → Add authorization rules in your security config.

678. What is CSRF protection in Spring Security?
     → A shield that stops sneaky cross-site attacks.

679. How do you disable CSRF for REST APIs?
     → Turn it off in the security config because REST usually doesn’t need it.

680. How do you implement role-based access control in Spring Boot?
     → Assign roles and lock endpoints with `hasRole()` or annotations.


### **Spring Boot Testing**

681. How do you test Spring Boot applications?
     → Use Spring’s testing support with mock tools and test slices.

682. What is `@SpringBootTest` used for?
     → Spinning up the whole app for full integration tests.

683. How do you test REST controllers?
     → Use `MockMvc` or `WebMvcTest`.

684. What is `@WebMvcTest`?
     → A test slice for just the web layer.

685. What is `@DataJpaTest`?
     → A test slice for JPA repositories with an in-memory DB.

686. How do you mock beans in Spring Boot tests?
     → Add `@MockBean` to replace real beans.

687. What is the difference between `@MockBean` and `@SpyBean`?
     → MockBean fakes everything; SpyBean wraps the real bean.

688. How do you perform integration testing in Spring Boot?
     → Use `@SpringBootTest` with real components wired together.

689. How do you use TestRestTemplate?
     → Auto-inject it in tests and call endpoints like a real client.

690. How do you use `MockMvc` for REST testing?
     → Build it with `@WebMvcTest` and send mock HTTP requests.


### **Advanced Spring Boot Features**

691. What are Spring Boot Starters?
     → Ready-made dependency bundles for common features.

692. What is the purpose of `spring-boot-starter-web`?
     → To build web and REST apps quickly.

693. What is the purpose of `spring-boot-starter-data-jpa`?
     → To plug in JPA and Spring Data easily.

694. What is the purpose of `spring-boot-starter-security`?
     → To add authentication and authorization support.

695. How do you enable asynchronous processing in Spring Boot?
     → Add `@EnableAsync` in a config class.

696. What is the `@Async` annotation used for?
     → Running a method in the background.

697. How do you configure task executors in Spring Boot?
     → Define a `TaskExecutor` bean with custom settings.

698. What is the difference between synchronous and asynchronous methods?
     → Sync waits for results; async runs independently.

699. How do you enable caching in Spring Boot?
     → Add `@EnableCaching` and use caching annotations.

700. How do you configure cache managers in Spring Boot?
     → Define a cache manager bean or use an auto-configured one.


---

## 🌐 **Batch 8 — Microservices, Spring Cloud, and Distributed Systems (Q701–Q800)**

### **Microservices Basics**

701. What is a microservices architecture?
     → A way of building apps as lots of tiny, independent services that work together.

702. How do microservices differ from a monolithic architecture?
     → Microservices break everything into small pieces, while monoliths keep everything glued together in one big chunk.

703. What are the benefits of microservices?
     → They make scaling, updating, and deploying parts of the system easier and safer.

704. What are the challenges of microservices?
     → Managing many moving parts, communication, and failures becomes trickier.

705. What is service decomposition?
     → Splitting a big system into small, focused services.

706. How do you identify microservice boundaries?
     → Look for natural business areas that can work on their own without stepping on others’ toes.

707. What is the difference between synchronous and asynchronous communication in microservices?
     → Synchronous waits for a reply; asynchronous sends a message and moves on happily.

708. What is the role of APIs in microservices?
     → APIs let services talk to each other politely and clearly.

709. What is API Gateway?
     → A single entry door that routes requests to the right microservice.

710. How does an API Gateway differ from a load balancer?
     → A gateway handles routing and extra logic, while a load balancer mainly spreads traffic evenly.

711. What is service discovery in microservices?
     → A system that helps services find each other without playing hide-and-seek.

712. What are some service discovery tools?
     → Tools like Consul, Eureka, and etcd.

713. What is client-side vs server-side discovery?
     → Client-side makes the caller pick the service; server-side lets a central helper choose.

714. What is circuit breaker pattern in microservices?
     → A safety switch that stops calling a failing service for a while.

715. Why is the circuit breaker pattern important?
     → It prevents one broken service from dragging everything else down.

716. What is the difference between a circuit breaker and a fallback?
     → A circuit breaker stops bad calls; a fallback provides a backup response instead.

717. What is load balancing in microservices?
     → Spreading requests across many service instances so none of them screams for help.

718. What are some strategies for load balancing?
     → Round robin, least connections, and random selection.

719. What is the role of configuration management in microservices?
     → It keeps all services using the right settings without chaos.

720. How do microservices handle failure and retries?
     → They retry safely, use timeouts, and lean on patterns like circuit breakers and backoff.


### **Spring Cloud Basics**

721. What is Spring Cloud?
     → A toolkit that helps build distributed, cloud-friendly microservices in the Spring world.

722. How does Spring Cloud complement Spring Boot?
     → Spring Boot makes apps easy; Spring Cloud adds tools to connect and manage them.

723. What are the main modules of Spring Cloud?
     → Config, Eureka, Gateway, LoadBalancer, Feign, Sleuth, and more.

724. What is Spring Cloud Config?
     → A system for managing all service configurations in one neat place.

725. How does Spring Cloud Config provide centralized configuration?
     → It stores configs in a shared repo and serves them to services on demand.

726. What is the role of the Config Server?
     → It fetches config files from the repo and hands them out to clients.

727. How do clients access configuration from Spring Cloud Config?
     → They call the Config Server automatically at startup or refresh time.

728. What is Spring Cloud Netflix Eureka?
     → A service registry that helps microservices find each other.

729. How does Eureka support service discovery?
     → Services register with it, and Eureka shares their locations with others.

730. What is a Eureka Client?
     → A service that registers itself with Eureka and looks up others.

731. How does a microservice register with Eureka?
     → It starts up with Eureka enabled and sends regular “I’m alive” updates.

732. What is Spring Cloud Gateway?
     → A modern gateway that routes, filters, and secures requests.

733. What are the differences between Zuul and Spring Cloud Gateway?
     → Gateway is faster, newer, reactive, and easier to extend than Zuul.

734. What is Spring Cloud Ribbon?
     → A tool for client-side load balancing.

735. How does Ribbon perform client-side load balancing?
     → It keeps a list of service instances and picks one based on a strategy.

736. What is the difference between Ribbon and Spring Cloud LoadBalancer?
     → Ribbon is older and deprecated; LoadBalancer is the newer replacement.

737. What is Spring Cloud Feign?
     → A declarative REST client that turns HTTP calls into simple interfaces.

738. How does Feign simplify REST client calls?
     → You just write an interface and Feign handles the calling magic.

739. How does Feign integrate with Ribbon for load balancing?
     → Feign uses Ribbon behind the scenes to pick a service instance.

740. What is Spring Cloud Hystrix?
     → A fault-tolerance tool that adds circuit breaking and resilience.


### **REST Best Practices**

741. What are some best practices for designing REST APIs?
742. What are RESTful principles?
743. What is the difference between PUT and PATCH?
744. When should you use POST vs PUT?
745. What is HATEOAS?
746. How do you implement versioning in REST APIs?
747. What are some common approaches to API versioning?
748. How do you handle exceptions in REST APIs?
749. How do you implement global exception handling in Spring Boot REST?
750. What HTTP status codes should you use for common scenarios?

### **Microservices Data Management**

741. What are some best practices for designing REST APIs?
     → Keep URLs clean, use proper HTTP methods, return clear status codes, and keep things predictable.

742. What are RESTful principles?
     → Use resources, standard HTTP verbs, stateless calls, and clear representations.

743. What is the difference between PUT and PATCH?
     → PUT replaces the whole thing; PATCH updates only the parts you care about.

744. When should you use POST vs PUT?
     → POST creates something new; PUT updates or replaces something that already exists.

745. What is HATEOAS?
     → A way to include helpful links in responses so clients know what actions to take next.

746. How do you implement versioning in REST APIs?
     → Add a version number in the URL, header, or media type.

747. What are some common approaches to API versioning?
     → URI versioning, header versioning, and content-type versioning.

748. How do you handle exceptions in REST APIs?
     → Catch errors, return friendly messages, and send proper status codes.

749. How do you implement global exception handling in Spring Boot REST?
     → Use `@ControllerAdvice` with `@ExceptionHandler`.

750. What HTTP status codes should you use for common scenarios?
     → 200 for success, 201 for created, 400 for bad input, 401 for unauthorized, 404 for not found, 500 for server errors.


### **Messaging & Asynchronous Communication**

761. What is the difference between synchronous and asynchronous messaging?
     → Synchronous waits for a reply; asynchronous fires the message and happily carries on.

762. What is a message broker?
     → A middleman that takes messages from one place and safely hands them to another.

763. What are some popular message brokers in Java ecosystem?
     → RabbitMQ, Kafka, ActiveMQ, and Redis Streams.

764. How do you integrate Spring Boot with RabbitMQ?
     → Use Spring AMQP starters and let Spring handle the queues and listeners for you.

765. How do you integrate Spring Boot with Kafka?
     → Use Spring Kafka starters and define producers, consumers, and topics.

766. What is the difference between a queue and a topic in messaging systems?
     → A queue sends each message to one consumer; a topic broadcasts it to everyone listening.

767. What is the role of producers and consumers?
     → Producers send messages; consumers receive and process them.

768. What is the difference between point-to-point and publish-subscribe messaging?
     → Point-to-point is one-to-one; publish-subscribe is one-to-many.

769. How do you ensure message delivery guarantees?
     → Use acknowledgements, retries, durable storage, and good old persistence.

770. What are idempotent consumers in messaging?
     → Consumers that can process the same message again without messing anything up.


### **Resiliency & Observability**

771. What is resiliency in microservices?
     → The ability of services to stay steady and recover gracefully when things go wrong.

772. How do you implement retries in microservices?
     → Use retry libraries or configs that automatically re-attempt failed calls with delays.

773. What is a fallback mechanism?
     → A backup response used when the main call fails.

774. What is rate limiting?
     → A way to control how many requests a client can make so the system doesn’t get overwhelmed.

775. How do you implement circuit breakers using Spring Cloud?
     → Use Spring Cloud Circuit Breaker with libraries like Resilience4j.

776. What is monitoring in microservices?
     → Watching services’ health, performance, and behaviour in real time.

777. What is logging in distributed systems?
     → Collecting and storing events from many services so issues can be understood later.

778. What is distributed tracing?
     → Tracking a request as it hops across multiple services.

779. What are some tools for distributed tracing?
     → Zipkin, Jaeger, and OpenTelemetry.

780. What is the difference between metrics, logs, and traces?
     → Metrics show numbers over time, logs record events, and traces follow request paths.


### **Security in Microservices**

781. What are the main security challenges in microservices?
     → Securing many small services, their communication, and their data all at once.

782. How do you secure REST APIs?
     → Use authentication, authorization, HTTPS, and proper validation.

783. What is OAuth2?
     → A framework that lets users grant limited access without sharing passwords.

784. What is JWT (JSON Web Token)?
     → A compact token that carries signed user data for secure stateless authentication.

785. How do you implement JWT authentication in Spring Boot microservices?
     → Generate tokens on login, validate them with filters, and secure routes with Spring Security.

786. What is API key-based authentication?
     → A simple method where clients include a unique key to prove who they are.

787. How do you handle user roles and permissions?
     → Assign roles and restrict access to routes based on those roles.

788. What is CORS, and how do you configure it in Spring Boot?
     → CORS controls which domains can call your API, set via `@CrossOrigin` or WebConfig.

789. How do you protect microservices against CSRF attacks?
     → Use CSRF tokens or simply disable CSRF when using stateless JWT APIs.

790. What is rate limiting and throttling for security?
     → Restricting how many requests a client can make to prevent abuse.


### **Spring Cloud Advanced Patterns**

791. What is a config server vs config client?
     → The server stores and serves configs; the client fetches and uses them.

792. How do Spring Cloud Bus and messaging work together?
     → The Bus uses a message broker to broadcast config changes across services.

793. What is Spring Cloud Sleuth?
     → A tool that adds tracing IDs to logs automatically.

794. How does Sleuth help with distributed tracing?
     → It tags every request with trace IDs so you can follow them across services.

795. What is Spring Cloud Zipkin?
     → A tracing system that collects and shows trace data from Sleuth.

796. How do you visualize distributed traces?
     → Open the Zipkin UI (or Jaeger) to see timelines and service hops.

797. What is Spring Cloud Consul?
     → A toolkit that uses Consul for discovery, config, and health checks.

798. What is the difference between Consul and Eureka?
     → Consul offers discovery plus KV storage; Eureka focuses mainly on discovery.

799. How do you perform blue-green deployments with microservices?
     → Run two versions side-by-side and switch traffic from old to new instantly.

800. How do you implement canary releases in microservices?
     → Send a small slice of traffic to the new version first, then increase gradually.


---

## ⚡ **Batch 9 — Advanced Java, JavaFX, Caching, Reactive, and Performance (Q801–Q900)**

### **Advanced Java Concepts**

801. What is reflection in Java?
     → A way for Java code to peek inside classes, fields, and methods while the program is running.

802. How do you obtain class information using reflection?
     → Grab a `Class` object using `Class.forName()` or `.getClass()` and inspect it.

803. How do you create objects dynamically with reflection?
     → Use `Class.newInstance()` or `clazz.getConstructor().newInstance()`.

804. How do you access private fields or methods via reflection?
     → Call `setAccessible(true)` on them before using them.

805. What are the use cases for reflection?
     → Framework magic like dependency injection, serialization, and runtime inspection.

806. What are the drawbacks of reflection?
     → It’s slower, less safe, and easy to mess things up if misused.

807. What are annotations in Java?
     → Tiny metadata tags that add extra meaning to code.

808. What are meta-annotations?
     → Special annotations that describe how other annotations behave.

809. How do you create a custom annotation?
     → Define it with `@interface` and specify its elements.

810. What is retention policy in annotations?
     → It decides how long an annotation sticks around (source, class, or runtime).

811. What is the difference between `@Retention(RetentionPolicy.RUNTIME)` and `CLASS`?
     → RUNTIME is kept alive for reflection; CLASS disappears at runtime.

812. How do you process annotations at runtime?
     → Use reflection to scan classes, methods, or fields for annotations.

813. What is the difference between marker, single-value, and multi-value annotations?
     → Marker has no values, single-value has one, multi-value has many.

814. What is the `@Repeatable` annotation?
     → It lets you slap the same annotation onto something multiple times.

815. How do annotations improve frameworks like Spring?
     → They let Spring auto-detect, configure, and wire things with less boilerplate.

816. What is the `Proxy` class in Java?
     → A utility that creates on-the-fly proxy objects for interfaces.

817. What is the difference between static and dynamic proxies?
     → Static ones are prewritten; dynamic ones are generated at runtime.

818. What is the role of `InvocationHandler`?
     → It intercepts method calls on a proxy and decides what happens.

819. What is the difference between Java SE dynamic proxy and CGLIB proxy?
     → Java SE proxies only wrap interfaces; CGLIB proxies subclass concrete classes.

820. How are proxies used in AOP?
     → They wrap target objects to add extra behaviors like logging or transactions.


### **JavaFX Basics**

821. What is JavaFX?
     → A modern Java library for building rich GUI applications with graphics, media, and UI controls.

822. How does JavaFX differ from Swing?
     → JavaFX is newer, supports CSS styling, FXML, and hardware-accelerated graphics; Swing is older and less flexible.

823. What are the main components of a JavaFX application?
     → `Stage`, `Scene`, and `Nodes`.

824. What is the `Stage` in JavaFX?
     → The top-level window of a JavaFX application.

825. What is the `Scene` in JavaFX?
     → The container holding all UI elements (Nodes) for display inside a Stage.

826. What are Nodes in JavaFX?
     → The basic building blocks of the UI like buttons, text fields, shapes, and layouts.

827. How do you create a simple UI with JavaFX?
     → Instantiate Nodes, add them to a layout, set the Scene on a Stage, and show it.

828. What are layouts in JavaFX?
     → Containers that arrange Nodes automatically, like `VBox`, `HBox`, or `GridPane`.

829. What is the difference between `VBox`, `HBox`, and `GridPane`?
     → `VBox` stacks vertically, `HBox` stacks horizontally, `GridPane` arranges in rows and columns.

830. How do you handle events in JavaFX?
     → Attach an `EventHandler` to Nodes or use lambda expressions.

831. What is the `EventHandler` interface?
     → It defines a single `handle(Event e)` method to process events.

832. How do you bind UI components to data?
     → Use properties and bind them with `.bind()` or `.bindBidirectional()`.

833. What is property binding in JavaFX?
     → Automatically synchronizing a UI component’s property with another property.

834. What is observable list in JavaFX?
     → A list that notifies listeners when its data changes, used for dynamic UI updates.

835. How do you style JavaFX components with CSS?
     → Assign a CSS file via `scene.getStylesheets().add()` and use style classes or IDs.

836. How do you load FXML files in JavaFX?
     → Use `FXMLLoader.load(getClass().getResource("file.fxml"))`.

837. What is the role of `FXMLLoader`?
     → It parses FXML, creates the UI, and wires it with the controller.

838. How do you handle controller classes in FXML?
     → Specify the controller in FXML with `fx:controller` and use `@FXML` to link components.

839. How do you create custom controls in JavaFX?
     → Extend existing controls or `Region` and define behavior, style, and layout.

840. What are animations and transitions in JavaFX?
     → Tools to change Node properties over time, like `FadeTransition`, `TranslateTransition`, and `Timeline`.


### **Caching**

841. What is caching in Java applications?
     → Temporarily storing frequently accessed data to improve performance.

842. What is the difference between in-memory and distributed cache?
     → In-memory cache is local to one JVM; distributed cache is shared across multiple servers.

843. What are popular caching frameworks in Java?
     → Ehcache, Caffeine, Hazelcast, and Redis.

844. What is Ehcache?
     → A Java in-memory caching library for boosting application performance.

845. What is Redis?
     → An open-source, in-memory key-value store often used as a cache or database.

846. How do you integrate Redis with Spring Boot?
     → Add Spring Data Redis dependency, configure RedisConnectionFactory, and use `@Cacheable`/`RedisTemplate`.

847. What is the difference between `@Cacheable` and `@CachePut`?
     → `@Cacheable` reads from cache if present; `@CachePut` always updates the cache.

848. What is `@CacheEvict` used for?
     → Removing entries from the cache.

849. What is the difference between read-through and write-through caching?
     → Read-through fetches data into cache on read miss; write-through updates cache when writing to DB.

850. What is cache invalidation, and why is it important?
     → Removing or updating stale cache entries to ensure data consistency.


### **Messaging & Patterns**

851. What is the publisher-subscriber pattern?
     → A design where publishers send messages to multiple subscribers without knowing them directly.

852. What is the difference between message queues and topics?
     → Queues deliver each message to one consumer; topics broadcast messages to all subscribers.

853. What is point-to-point messaging?
     → A messaging model where a message is sent to a specific queue and consumed by a single receiver.

854. How do you ensure exactly-once message delivery?
     → Combine idempotent processing, deduplication, and transactional messaging.

855. What is the difference between at-most-once, at-least-once, and exactly-once delivery?
     → At-most-once: may lose messages; At-least-once: may duplicate; Exactly-once: delivered once reliably.

856. What is a dead letter queue?
     → A queue that holds messages that cannot be delivered or processed successfully.

857. How do you handle message retries?
     → Retry with backoff, limit attempts, and move failed messages to a dead letter queue.

858. What is idempotency in messaging?
     → Processing a message multiple times has the same effect as processing it once.

859. What is a fan-out pattern in messaging?
     → A single message is sent to multiple consumers simultaneously.

860. How do microservices communicate asynchronously using messaging?
     → By publishing messages to queues or topics, letting services process them independently.


### **Reactive Programming**

861. What is reactive programming?
     → A programming paradigm focused on asynchronous data streams and the propagation of change.

862. What is the difference between reactive and imperative programming?
     → Imperative tells *how* to do things step by step; reactive declares *what* happens when data changes.

863. What are the main principles of reactive programming?
     → Responsive, resilient, elastic, and message-driven.

864. What is backpressure in reactive streams?
     → A mechanism to prevent a fast producer from overwhelming a slow consumer.

865. What are the core interfaces in Project Reactor?
     → `Publisher`, `Subscriber`, `Subscription`, `Processor`, plus Reactor’s `Mono` and `Flux`.

866. What is a `Mono`?
     → A reactive stream that emits 0 or 1 item.

867. What is a `Flux`?
     → A reactive stream that emits 0 to N items.

868. How do you create a reactive pipeline using Reactor?
     → Chain operators like `map()`, `flatMap()`, `filter()`, and `subscribe()` on a `Mono` or `Flux`.

869. What is `subscribeOn()` vs `publishOn()`?
     → `subscribeOn()` changes the thread where subscription happens; `publishOn()` changes the thread for downstream operations.

870. How does reactive programming integrate with Spring WebFlux?
     → WebFlux uses reactive types (`Mono`, `Flux`) to handle HTTP requests non-blockingly.

871. What is the difference between Spring MVC and WebFlux?
     → MVC is blocking and servlet-based; WebFlux is non-blocking and reactive.

872. How do you handle exceptions in reactive pipelines?
     → Use operators like `onErrorReturn()`, `onErrorResume()`, or `doOnError()`.

873. What is the role of `Schedulers` in Reactor?
     → They control which thread or thread pool executes reactive operations.

874. What is the difference between hot and cold publishers?
     → Cold emits data per subscriber; hot emits data regardless of subscribers.

875. How do you combine multiple reactive streams?
     → Use operators like `merge()`, `zip()`, `concat()`, or `combineLatest()`.

876. What is the difference between `flatMap()` and `map()` in reactive streams?
     → `map()` transforms items synchronously; `flatMap()` transforms into another reactive stream asynchronously.

877. How do you test reactive code?
     → Use `StepVerifier` from Reactor Test to verify emitted sequences and behaviors.

878. What is the difference between blocking and non-blocking IO in reactive applications?
     → Blocking waits for results; non-blocking returns immediately and handles results asynchronously.

879. How does reactive programming improve scalability?
     → By freeing threads from waiting, allowing the system to handle many concurrent tasks efficiently.

880. What is reactive backpressure strategy?
     → Techniques like buffering, dropping, or throttling to manage fast producers and slow consumers.


### **Performance Tuning & Optimization**

881. What are common JVM tuning options?
     → `-Xmx`, `-Xms`, `-XX:+UseG1GC`, `-XX:MaxGCPauseMillis`, `-XX:+HeapDumpOnOutOfMemoryError`.

882. How do you analyze memory usage in Java?
     → Use tools like VisualVM, JConsole, or Java Flight Recorder to inspect heap and non-heap memory.

883. What is the difference between heap and non-heap memory?
     → Heap stores objects; non-heap stores metadata, code, and JVM internal structures.

884. What is the difference between minor GC and major GC?
     → Minor GC cleans young generation; major GC cleans old generation and is slower.

885. How do you choose the right garbage collector?
     → Based on throughput, latency requirements, and application memory footprint.

886. What is the difference between Serial, Parallel, CMS, and G1 garbage collectors?
     → Serial: single-threaded; Parallel: multi-threaded; CMS: concurrent low-pause; G1: concurrent with region-based heap.

887. How do you optimize thread pools?
     → Right-size threads, avoid blocking calls, and use `Executors` wisely.

888. How do you detect thread contention?
     → Monitor thread dumps, use profiling tools, or watch for high CPU and thread-blocked states.

889. How do you profile a Java application?
     → Use profiling tools to track memory, CPU, and thread activity during runtime.

890. What tools can you use for profiling (memory, CPU)?
     → VisualVM, YourKit, JProfiler, Java Flight Recorder.

891. What are JVM flags for performance tuning?
     → `-Xmx`, `-Xms`, `-XX:+UseG1GC`, `-XX:+UseParallelGC`, `-XX:MaxGCPauseMillis`.

892. How do you avoid memory leaks in Java?
     → Remove unused references, close resources, and use weak references when appropriate.

893. How do you minimize object creation overhead?
     → Reuse objects, use primitives when possible, and avoid unnecessary temporary objects.

894. What is object pooling?
     → Reusing objects from a pool instead of creating new ones each time.

895. How do you reduce GC pauses in large applications?
     → Use concurrent or low-pause GCs, tune heap sizes, and minimize object churn.

896. How do you optimize SQL queries in Java applications?
     → Use indexing, prepared statements, batch operations, and avoid N+1 query problems.

897. How do you reduce network latency in microservices?
     → Use caching, asynchronous calls, connection pooling, and minimize payload sizes.

898. What is connection pooling, and why is it important?
     → Reusing database/network connections to reduce creation overhead and improve performance.

899. How do you implement caching for performance optimization?
     → Use in-memory caches (Ehcache, Caffeine) or distributed caches (Redis) for frequently accessed data.

900. How do you measure application performance in production?
     → Use APM tools (New Relic, Dynatrace), logs, metrics, and real-time monitoring dashboards.


---

## 🏁 **Batch 10 — Advanced Microservices, Security, Testing, DevOps, Cloud, and Java 17+ (Q901–Q1000)**

### **Advanced Spring Microservices**

901. What is the difference between monolithic and microservices in terms of deployment?
     → Monolithic: single deployable unit; Microservices: multiple independently deployable services.

902. How do you implement inter-service communication?
     → Use REST, gRPC, or messaging systems like Kafka/RabbitMQ.

903. What is API composition vs Command Query Responsibility Segregation (CQRS) in microservices?
     → API composition: gather data from multiple services for a client; CQRS: separate read and write operations for scalability.

904. How do you handle distributed transactions in microservices?
     → Use Saga patterns or two-phase commit for consistency across services.

905. What is Spring Cloud Gateway used for?
     → It’s an API gateway for routing, filtering, and securing requests to microservices.

906. How do you implement rate limiting in Spring Cloud Gateway?
     → Configure `RequestRateLimiter` filter with Redis or in-memory policies.

907. How do you implement authentication and authorization in Spring Cloud Gateway?
     → Use filters that validate JWT tokens or OAuth2 tokens before forwarding requests.

908. What is the difference between OAuth2 and OpenID Connect?
     → OAuth2: authorization framework; OpenID Connect: authentication layer built on OAuth2.

909. How do you secure microservices with JWT tokens?
     → Issue JWTs from auth service and verify them in each microservice before processing requests.

910. How do you implement API versioning in microservices?
     → Use URL path, request header, or query parameter versioning strategies.


### **Spring Cloud & Resiliency**

911. What is Spring Cloud Circuit Breaker?
     → A library to gracefully handle failures in microservices by stopping cascading errors.

912. How do you implement fallback methods in Spring Cloud?
     → Annotate with `@CircuitBreaker(name="service", fallbackMethod="methodName")`.

913. What is bulkhead pattern in microservices?
     → Isolates resources so one failing service doesn’t crash others.

914. What is retry pattern?
     → Automatically re-attempts failed operations a set number of times before failing.

915. What is timeout pattern?
     → Limits how long a service waits for a response to avoid hanging calls.

916. How do you implement caching in microservices using Redis?
     → Use `Spring Cache` with `@Cacheable` and configure Redis as the cache manager.

917. What is Spring Cloud Sleuth used for?
     → Adds tracing information to logs to track requests across microservices.

918. How does Sleuth integrate with Zipkin for tracing?
     → Sleuth sends trace data to Zipkin, which visualizes the call chains and latencies.

919. How do you monitor microservices using Prometheus and Grafana?
     → Expose metrics via `/actuator/prometheus` and visualize them in Grafana dashboards.

920. How do you implement distributed logging in microservices?
     → Use centralized logging systems like ELK stack or Loki, collecting logs from all services.


### **Security & Identity**

921. How do you implement single sign-on (SSO) in microservices?
     → Use a centralized identity provider (like Keycloak or Okta) to authenticate users across all services.

922. How do you secure REST APIs with Spring Security OAuth2?
     → Configure OAuth2 Resource Server to validate access tokens before granting access.

923. What is the difference between symmetric and asymmetric encryption?
     → Symmetric: same key for encrypt/decrypt; Asymmetric: public key encrypts, private key decrypts.

924. How do you store secrets securely in microservices?
     → Use secret management tools like Vault, AWS Secrets Manager, or environment variables encrypted at rest.

925. How do you implement refresh tokens in OAuth2?
     → Issue refresh tokens along with access tokens and use them to get new access tokens when expired.

926. What is the difference between session-based and token-based authentication?
     → Session-based stores state on server; token-based (like JWT) stores state in client token.

927. What is Spring Security’s filter chain?
     → A sequence of filters that intercept requests for authentication, authorization, and other security checks.

928. How do you handle Cross-Origin Resource Sharing (CORS) in Spring Boot?
     → Configure allowed origins, methods, and headers via `@CrossOrigin` or `WebMvcConfigurer`.

929. How do you prevent CSRF attacks in REST APIs?
     → Disable CSRF for stateless APIs or use CSRF tokens for stateful endpoints.

930. How do you implement role-based access control (RBAC)?
     → Define roles and use `@PreAuthorize` or `hasRole()` checks in controllers/services.


### **Testing & CI/CD**

931. What is unit testing vs integration testing in Spring Boot?
     → Unit testing: tests individual components in isolation; Integration testing: tests how components work together.

932. What is `@SpringBootTest` used for?
     → Boots up the full application context for integration tests.

933. How do you test REST endpoints with `MockMvc`?
     → Use `MockMvc` to perform HTTP requests and assert responses without starting the server.

934. How do you test JPA repositories with `@DataJpaTest`?
     → Loads only JPA components and an in-memory database for repository testing.

935. What is test slicing in Spring Boot?
     → Loads only relevant parts of the application context to speed up tests.

936. How do you use `TestRestTemplate` for integration tests?
     → Send real HTTP requests to the running server and verify responses.

937. How do you mock beans in Spring tests?
     → Use `@MockBean` to replace a bean with a mock in the application context.

938. What is the difference between `@MockBean` and `@SpyBean`?
     → `@MockBean`: creates a mock; `@SpyBean`: wraps real bean allowing partial mocking.

939. How do you write contract tests for microservices?
     → Use tools like Pact or Spring Cloud Contract to verify that services adhere to agreed APIs.

940. What is the role of CI/CD in microservices?
     → Automates build, test, and deployment pipelines for faster and reliable releases.

941. How do you integrate Spring Boot tests with Jenkins or GitHub Actions?
     → Configure pipelines to run `mvn test` or `gradle test` as part of CI steps.

942. How do you perform automated deployment to Kubernetes or cloud platforms?
     → Use CI/CD pipelines with `kubectl`, Helm charts, or cloud provider deployment APIs.

943. What is canary deployment, and how is it implemented?
     → Gradually releases a new version to a small subset of users before full rollout.

944. What is blue-green deployment in microservices?
     → Maintains two identical environments; traffic is switched from old (blue) to new (green) version.

945. How do you implement rollback strategies for failed deployments?
     → Keep previous versions ready and switch back traffic if the new version fails.

946. How do you perform performance testing for microservices?
     → Use tools like JMeter, Gatling, or Locust to simulate load and measure metrics.

947. What is the difference between load testing, stress testing, and endurance testing?
     → Load: normal/high load; Stress: beyond capacity; Endurance: long-term stability.

948. How do you measure response time and throughput in production?
     → Collect metrics via APM tools, logs, or monitoring dashboards.

949. What tools are commonly used for monitoring performance?
     → Prometheus, Grafana, New Relic, Datadog, ELK stack.

950. How do you monitor JVM metrics in production?
     → Use JMX, Micrometer, or Prometheus exporters to track memory, GC, threads, and CPU.


### **Cloud & Containerization**

951. How do you containerize a Spring Boot application?
     → Write a `Dockerfile` to package the JAR and dependencies, then build and run the Docker image.

952. What is Docker, and how does it help with microservices?
     → Docker is a container platform that packages apps with all dependencies for consistent deployment.

953. What is the difference between a Docker image and a container?
     → Image: a blueprint; Container: a running instance of that image.

954. How do you run multiple Spring Boot services using Docker Compose?
     → Define services in `docker-compose.yml` and run `docker-compose up`.

955. What is Kubernetes?
     → A container orchestration platform for deploying, scaling, and managing containers.

956. How do you deploy Spring Boot microservices to Kubernetes?
     → Create Deployment and Service YAML files, then apply them with `kubectl apply -f`.

957. What are pods, deployments, and services in Kubernetes?
     → Pod: smallest deployable unit; Deployment: manages pod replicas; Service: exposes pods for access.

958. How do you perform scaling in Kubernetes?
     → Use `kubectl scale` or set `replicas` in Deployment YAML; Horizontal Pod Autoscaler can automate it.

959. How do you implement service discovery in Kubernetes?
     → Kubernetes Services provide DNS names to locate and connect pods automatically.

960. How do you perform logging and monitoring in Kubernetes?
     → Use centralized logging (ELK, Loki) and monitoring (Prometheus, Grafana) via sidecars or agents.

961. What is Helm, and how does it help with deployment?
     → Helm is a package manager for Kubernetes that simplifies deploying and managing apps with charts.

962. What is the difference between stateful and stateless services in the cloud?
     → Stateless: no persistent state; Stateful: maintains data or session across requests.

963. How do you implement configuration management in cloud environments?
     → Use ConfigMaps, environment variables, or external config servers like Spring Cloud Config.

964. How do you handle secrets in Kubernetes?
     → Store sensitive data in `Secrets` and mount them as environment variables or volumes.

965. What is cloud-native design?
     → Building applications to fully leverage cloud features: scalability, resilience, and automation.

966. How do microservices achieve high availability in cloud environments?
     → Deploy multiple instances across zones, use load balancers, and implement failover strategies.

967. What is a service mesh?
     → An infrastructure layer that manages service-to-service communication, security, and observability.

968. How does Istio integrate with Spring Boot microservices?
     → Injects sidecar proxies to manage traffic, security, and telemetry without changing service code.

969. How do you implement traffic routing and observability with Istio?
     → Use VirtualServices and DestinationRules for routing; collect metrics, logs, and traces via telemetry.

970. What are best practices for cloud-native microservices?
     → Design for resilience, scalability, observability, statelessness, automated CI/CD, and secure defaults.


### **Java 17+ Features**

971. What are records in Java 16/17?
     → Special classes for immutable data with auto-generated constructors, getters, `equals`, `hashCode`, and `toString`.

972. How do records differ from normal classes?
     → Records are concise, immutable, and primarily store data, unlike regular classes which require boilerplate code.

973. What are sealed classes in Java 17?
     → Classes or interfaces that restrict which other classes can extend or implement them.

974. How do sealed classes enhance OOP design?
     → They enforce controlled inheritance, improving type safety and maintainability.

975. What is pattern matching for `instanceof`?
     → Allows type casting inline when checking an object’s type.

976. What is pattern matching for `switch` in Java 17+?
     → Lets `switch` directly match types and bind variables without explicit casting.

977. What are text blocks, and how do they simplify multiline strings?
     → Multiline string literals using `"""` that preserve formatting and reduce escaping.

978. What is the difference between `var` and explicit typing?
     → `var` infers type from the initializer; explicit typing declares the type explicitly.

979. What is a `sealed interface`?
     → An interface that restricts which classes can implement it using `permits`.

980. What are the new collection factory methods in Java 9+?
     → `List.of()`, `Set.of()`, and `Map.of()` to create immutable collections concisely.

981. How do you use `Optional.or()` and `Optional.stream()`?
     → `or()`: fallback Optional; `stream()`: converts Optional to a Stream for fluent processing.

982. What is `Stream.toList()` in Java 16+?
     → Collects stream elements into an immutable List.

983. What are helpful enhancements in `Map` and `Set` in Java 17?
     → New `Map.copyOf()`, `Set.of()` factory methods, and `Map.computeIfAbsent` improvements.

984. What is the difference between `==` and `equals()` for records?
     → `==` checks reference equality; `equals()` checks content equality, auto-implemented for records.

985. What is switch expression vs switch statement?
     → Switch expression returns a value; switch statement executes statements without returning.

986. How do you use `instanceof` pattern binding?
     → `if (obj instanceof String s) { … }` binds `obj` to `s` directly if the type matches.

987. What are enhanced pseudo-random number generators (PRNG) in Java 17?
     → New `RandomGenerator` interface and algorithms with better performance and security options.

988. What is the difference between `sealed`, `non-sealed`, and `permits`?
     → `sealed`: restricts subclassing; `non-sealed`: allows unrestricted subclassing; `permits`: lists allowed subclasses.

989. How do you use pattern matching to simplify code?
     → Eliminates explicit casting and conditional checks when processing different object types.

990. What are the key language enhancements in Java 17 LTS?
     → Records, sealed classes, pattern matching, text blocks, new switch expressions, enhanced PRNGs, and collection factory methods.


### **Enterprise Patterns & Best Practices**

991. What is CQRS in enterprise architecture?
     → Command Query Responsibility Segregation: separates read (query) and write (command) operations for scalability and clarity.

992. How does Event Sourcing work?
     → Stores state changes as a sequence of immutable events instead of overwriting current state.

993. What is the Saga pattern in distributed transactions?
     → A sequence of local transactions with compensating actions to maintain consistency across services.

994. What is eventual consistency?
     → Data will become consistent across distributed systems over time, without immediate synchronization.

995. What are distributed locks, and how do you implement them?
     → Mechanism to control access to shared resources across services; use Redis, Zookeeper, or database locks.

996. How do you handle idempotent operations in microservices?
     → Design operations so repeated requests produce the same effect without side effects.

997. What are anti-corruption layers?
     → A layer that translates between legacy systems and new systems to prevent design contamination.

998. How do you implement the strangler pattern for migration?
     → Gradually replace parts of a legacy system with new services, routing traffic incrementally.

999. How do you ensure observability with logs, metrics, and tracing?
     → Collect structured logs, expose metrics, and trace requests across services using tools like ELK, Prometheus, and Zipkin.

1000. What are best practices for building cloud-native, resilient, and maintainable Java microservices?
      → Design for statelessness, scalability, observability, CI/CD automation, fault tolerance, secure defaults, and clear service boundaries.

---

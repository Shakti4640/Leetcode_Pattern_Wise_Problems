Below are **100 one-liner hard interview questions**, each with a **tiny hint** to jog the mind.

---

# ✅ **Batch 1 — Java Basics & OOP (Q1–Q100)**

### **One-Line Hard Questions + Tiny Hints**

1. What exactly happens in memory when you create a new object?
   → Hint: heap party + stack directions

2. Why is `String` immutable in Java?
   → Hint: safety + caching + class loader peace

3. What does the JVM actually do when a program starts?
   → Hint: loads, checks, runs… and babysits threads

4. Explain how autoboxing can secretly hurt performance.
   → Hint: tiny objects everywhere

5. Why is `float` not precise?
   → Hint: binary can’t handle some decimals nicely

6. What does the `final` keyword actually guarantee?
   → Hint: no change… but depends on where you put it

7. How does the `==` operator behave differently for primitives vs objects?
   → Hint: values vs addresses

8. What really happens during method overloading resolution?
   → Hint: compiler plays match-maker

9. What is the purpose of a class loader hierarchy?
   → Hint: parents first, kids later

10. What happens if a constructor throws an exception?
    → Hint: object never gets born

11. Why can’t you override static methods?
    → Hint: they belong to the class, not you

12. What exactly is a marker interface?
    → Hint: a tag with no mouth but serious power

13. Why does Java enforce single inheritance?
    → Hint: diamond problems make languages cry

14. What does the `volatile` keyword do at the memory level?
    → Hint: stop caching tricks

15. How does Java determine which overloaded method to call with `null`?
    → Hint: most specific wins… usually

16. What’s the difference between a shallow copy and a deep copy?
    → Hint: twins vs clones

17. How does dynamic binding actually work?
    → Hint: JVM decides at the last second

18. What is the role of a class’s constant pool?
    → Hint: storage for literal goodies

19. Why can interfaces have default methods now?
    → Hint: evolution without breaking elders

20. How does Java resolve multiple interface default method conflicts?
    → Hint: you must choose the favourite child

21. What happens if you call `this()` and `super()` together?
    → Hint: compiler says NOPE

22. Why can enums have methods?
    → Hint: they’re classes in fancy uniforms

23. Explain method hiding with static methods.
    → Hint: not overriding… just pretending

24. Why is the main method static?
    → Hint: JVM needs to start somewhere without setup

25. What is the real difference between `StringBuilder` and `StringBuffer`?
    → Hint: speed vs safety

26. How does the compiler find unreachable code?
    → Hint: flow-chart brain

27. What does the JVM do when an exception is thrown?
    → Hint: starts climbing the call stack mountain

28. Why should you override `hashCode()` when overriding `equals()`?
    → Hint: collection happiness

29. How does Java prevent memory leaks without pointers?
    → Hint: GC + reference tracking

30. What is type erasure and why is it needed?
    → Hint: generics didn’t exist before

31. What happens when a thread dies inside a `finally` block?
    → Hint: chaos + unpredictable sadness

32. Why can abstract classes have constructors?
    → Hint: parents need setup too

33. How does Java treat an array under the hood?
    → Hint: it’s an object in disguise

34. Why is multiple public classes not allowed in a single file?
    → Hint: filename loyalty

35. What does the `super` keyword *really* reference?
    → Hint: parent’s hidden toolbox

36. Why should exceptions be immutable?
    → Hint: thread safety + debugging sanity

37. What is a class literal (`ClassName.class`)?
    → Hint: ticket to reflection land

38. How do wrapper classes maintain caching for some numbers?
    → Hint: −128 to 127 VIP section

39. What is the difference between checked and unchecked exceptions at bytecode level?
    → Hint: compiler vs JVM responsibility

40. Why are arrays covariant?
    → Hint: legacy decisions… not always good ones

41. Why are generics invariant?
    → Hint: safety first

42. How does Java detect stack overflow?
    → Hint: no more stack frames available

43. Why can’t constructors be inherited?
    → Hint: each class builds itself

44. What happens when you clone an object using `clone()`?
    → Hint: shallow magic by default

45. What exactly is the default value of uninitialized variables?
    → Hint: depends where they live

46. Why does `String + int` compile but `Object + int` doesn’t?
    → Hint: special treatment for strings

47. What causes the “diamond problem” in inheritance?
    → Hint: two paths, one confused child

48. How does Java represent a `null` reference internally?
    → Hint: special zero value

49. Why does Java use UTF-16 for `String`?
    → Hint: Unicode history baggage

50. How does iteration over a string work?
    → Hint: char array marching

51. Why do primitive arrays perform so well?
    → Hint: tight memory layout

52. What is the difference between `return;` and `return voidVar;`?
    → Hint: compiler behaviour

53. What happens when you divide by zero for integers vs floating numbers?
    → Hint: exception vs infinity

54. What is an anonymous inner class used for?
    → Hint: quick one-time helpers

55. Why can nested classes access private members of outer classes?
    → Hint: compiler creates bridges

56. What does “pass-by-value” really mean in Java?
    → Hint: copies of references

57. Why are logical operators short-circuited?
    → Hint: speed + safety

58. Why do `equals()` checks start with `==`?
    → Hint: quick escape route

59. What happens if a static block throws an exception?
    → Hint: class will refuse to load

60. Why use `transient`?
    → Hint: skip unwanted baggage during serialization

61. What is the difference between JDK, JRE, and JVM?
    → Hint: toolkit, kitchen, stove

62. Why is `main` not required in every Java program?
    → Hint: frameworks call the shots

63. How does Java handle integer overflow?
    → Hint: wrap around like a loop-the-loop

64. What is the role of the method area?
    → Hint: class data sleeping spot

65. Why use composition over inheritance?
    → Hint: more flexible family tree

66. What is the difference between `throw` and `throws`?
    → Hint: action vs declaration

67. How does the compiler treat switch statements with strings?
    → Hint: hash + equals magic

68. Why do enums prevent duplicate instances?
    → Hint: private constructor lockdown

69. What is a synthetic method?
    → Hint: compiler’s secret helpers

70. What happens when `System.gc()` is called?
    → Hint: polite request, not an order

71. Why can't you use primitives in generics?
    → Hint: type erasure limitations

72. How does Java ensure thread safety for class initialization?
    → Hint: synchronized behind the scenes

73. What is the difference between `instanceof` and `getClass()` checks?
    → Hint: inheritance vs identity

74. Why are static variables shared across objects?
    → Hint: they live in class land, not object land

75. How does the compiler inline constant values?
    → Hint: replaces them at compile time

76. What is the difference between JIT and AOT compilation?
    → Hint: now vs before

77. What happens during garbage collection of cyclic references?
    → Hint: GC doesn’t care about circles

78. Why does Java forbid multiple default constructors?
    → Hint: only one no-arg entry point

79. How does the stack frame store method parameters?
    → Hint: one tidy frame per call

80. What is a phantom reference?
    → Hint: ghost signals after GC

81. Why do strings use a hash cache?
    → Hint: speed boost for repeated lookups

82. What is a class initialization lock?
    → Hint: safe one-time setup

83. Why can an interface extend multiple interfaces?
    → Hint: behaviour mix-and-match

84. What happens when an exception is thrown in a static block?
    → Hint: class dies instantly

85. What is escape analysis?
    → Hint: JVM decides heap vs stack

86. Why does Java allow overloading but not overriding of private methods?
    → Hint: private members aren’t visible

87. How does the JVM verify bytecode?
    → Hint: safety checks like airport security

88. Why is `Object` the default root class?
    → Hint: universal parent

89. What is the difference between `intern()` and regular strings?
    → Hint: shared pool vs heap copies

90. Why are wrapper classes immutable?
    → Hint: caching + thread safety

91. What does `super()` always have to be the first line?
    → Hint: parents get setup first

92. How do multidimensional arrays work internally?
    → Hint: array of arrays, not a grid

93. Why can enums implement interfaces?
    → Hint: behaviour mix allowed

94. What is the difference between `char` and `Character`?
    → Hint: primitive vs dressed-up version

95. How does the JVM detect unreachable objects?
    → Hint: reference graph walk

96. Why is reflection expensive?
    → Hint: lots of checks and indirection

97. What is the difference between heap and metaspace?
    → Hint: objects vs class metadata

98. Why should you avoid using finalizers?
    → Hint: unpredictable zombie behaviour

99. What does the class loader do during linking?
    → Hint: verify + prepare + resolve

100. What is the difference between a no-arg constructor and the default constructor?
     → Hint: one exists only if you don’t make your own

---

### ✔️ **Batch 1 completed!**

---

# ✅ **Batch 2 — Core Java Advanced & Collections (Q101–Q200)**

### **One-Line Hard Questions + Tiny Hints**

101. How does `HashMap` actually calculate a bucket index?
      → Hint: hash → spread → index

102. Why does `HashMap` use power-of-two bucket sizes?
      → Hint: bitwise shortcuts

103. What problem does hash spreading solve?
      → Hint: avoids bucket crowding

104. What is the difference between hash collision and hash clustering?
      → Hint: same slot vs near slots

105. Why did Java 8 add tree bins to HashMap?
      → Hint: worst-case performance rescue

106. When does a HashMap bucket convert to a tree?
      → Hint: threshold + comparable keys

107. Why must keys used in HashMap be immutable?
      → Hint: stability of hash + equals

108. What happens when a key’s hash changes after insertion?
      → Hint: map can't find its own kid

109. How does a `HashSet` internally work?
      → Hint: it’s a HashMap in disguise

110. Why does `LinkedHashMap` maintain insertion order?
      → Hint: uses a double-linked list

111. How does LRU mode work in LinkedHashMap?
      → Hint: access order shuffle

112. Why is `TreeMap` slower than `HashMap`?
      → Hint: sorting costs time

113. What is the structure used by TreeMap?
      → Hint: Red-Black Tree

114. Why do TreeMap keys need to be Comparable or have a Comparator?
      → Hint: sorted tree needs directions

115. How does TreeSet enforce uniqueness?
      → Hint: compares, not hashes

116. What is the fail-fast behaviour of iterators?
      → Hint: detects sneaky changes

117. Why are fail-fast iterators not guaranteed?
      → Hint: best effort only

118. How does a ConcurrentHashMap avoid locking the entire map?
      → Hint: finer-grain control

119. Why does ConcurrentHashMap disallow null keys?
      → Hint: ambiguity with missing values

120. How does computeIfAbsent work internally?
      → Hint: double-check pattern

121. What is the difference between Enumeration and Iterator?
      → Hint: old vs newer

122. What is the role of Spliterator?
      → Hint: parallel-friendly iterator

123. Why is CopyOnWriteArrayList good for read-heavy situations?
      → Hint: snapshot magic

124. Why is CopyOnWriteArrayList terrible for write-heavy?
      → Hint: copying… every… time

125. How does WeakHashMap work?
      → Hint: keys disappear when GC runs

126. What’s the purpose of IdentityHashMap?
      → Hint: compares with `==`

127. Why would someone use ArrayDeque instead of Stack?
      → Hint: newer and faster

128. What is the time complexity of getting an element from ArrayList?
      → Hint: O(1) happiness

129. Why is LinkedList slow for random access?
      → Hint: must walk nodes

130. Why does ArrayList grow by 1.5x?
      → Hint: balance between space and speed

131. How does `ensureCapacity()` work?
      → Hint: pre-reserve seats

132. Why shouldn’t you modify a list while iterating without iterator methods?
      → Hint: fail-fast chaos

133. What happens if you call remove() twice in an iterator?
      → Hint: iterator scolds you

134. What is structural modification in collections?
      → Hint: anything that reshapes the container

135. How does PriorityQueue order elements?
      → Hint: heap structure

136. Why is PriorityQueue not a sorted queue?
      → Hint: only top element matters

137. What is the difference between comparator and comparable?
      → Hint: external judge vs built-in judge

138. Why must comparator logic be consistent with equals?
      → Hint: avoid weird behaviour

139. Why does Optional discourage null return values?
      → Hint: clear intent, less surprise

140. How does Optional.map differ from flatMap?
      → Hint: nested vs flattened results

141. Why are streams lazy?
      → Hint: do work only when needed

142. What triggers a stream pipeline to run?
      → Hint: terminal operation

143. Why are parallel streams dangerous?
      → Hint: shared state + randomness

144. How does a stream perform short-circuiting operations?
      → Hint: stops when happy

145. What is the difference between intermediate and terminal operations?
      → Hint: setup vs execution

146. What does the collector "toMap" do when keys collide?
      → Hint: needs merge instructions

147. Why can streams only be used once?
      → Hint: they close shop after running

148. How does reduce() differ from collect()?
      → Hint: fold vs container-build

149. Why are lambda expressions stateless by design?
      → Hint: helps parallelism

150. Why is method reference considered syntactic sugar?
      → Hint: shortcut for lambda

151. What is the difference between peek() and map()?
      → Hint: sneak vs transform

152. Why is Stream.ofNullable useful?
      → Hint: skip null inputs gracefully

153. How does groupingBy() work internally?
      → Hint: map builder

154. What is backpressure in stream-like systems?
      → Hint: too-fast producer meets slow consumer

155. How does LocalDate avoid mutability problems?
      → Hint: returns new objects

156. Why do date/time classes use ISO-8601?
      → Hint: universal standard

157. What is the difference between Period and Duration?
      → Hint: human vs machine time

158. Why is ZoneId essential for correct time calculations?
      → Hint: world is big

159. What is the difference between OffsetDateTime and ZonedDateTime?
      → Hint: fixed offset vs full rules

160. Why is EnumSet so fast?
      → Hint: bitwise operations

161. What is an iterator’s fail-safe behaviour?
      → Hint: works on copies

162. Why does TreeMap not allow null keys?
      → Hint: comparator can't compare nothing

163. What is a modCount?
      → Hint: collection’s change counter

164. Why should you choose ArrayList over Vector?
      → Hint: Vector synchronizes too much

165. How does a LinkedHashSet maintain order?
      → Hint: linked list inside

166. Why is BigInteger immutable?
      → Hint: thread-safety and predictability

167. Why is BigDecimal preferred for money?
      → Hint: accuracy > speed

168. What is the scale in BigDecimal?
      → Hint: digits after decimal

169. Why can BigDecimal comparisons be tricky?
      → Hint: scale differences

170. What is the purpose of NavigableMap?
      → Hint: floor, ceiling, nearby searches

171. Why are queues FIFO by design?
      → Hint: first come, first served

172. How does LinkedList behave as a queue?
      → Hint: head-tail operations

173. Why are Deques double-ended?
      → Hint: flexibility for stacks and queues

174. How does `forEachRemaining()` work?
      → Hint: bulk-visit all leftovers

175. Why does `Arrays.asList()` produce a fixed-size list?
      → Hint: wrapper around array

176. How does `Collections.unmodifiableList()` work?
      → Hint: wrapper that rejects edits

177. Why shouldn’t you rely on concurrent modification exceptions?
      → Hint: no guarantee

178. What is the difference between map.merge and compute?
      → Hint: merge is value-focused

179. Why are ConcurrentSkipListMap and ConcurrentSkipListSet useful?
      → Hint: sorted + concurrent

180. How does a skip list achieve O(log n)?
      → Hint: multi-level linked lists

181. Why does HashMap allow null keys but Hashtable doesn't?
      → Hint: old design vs new design

182. What is the resizing cost in HashMap?
      → Hint: rehashing storm

183. Why does ArrayDeque disallow null elements?
      → Hint: null is a special marker

184. What is a spliterator characteristic?
      → Hint: hints like ORDERED, SORTED

185. Why is identity-based hashing rare?
      → Hint: usually want equality

186. Why does Optional.orElseThrow exist?
      → Hint: fail loudly when necessary

187. Why does Optional discourage get()?
      → Hint: risky if empty

188. What is the collector "partitioningBy" used for?
      → Hint: true/false buckets

189. Why are method references faster to read?
      → Hint: shorthand for lambdas

190. What is a stream pipeline?
      → Hint: staged operations

191. Why is map multi-thread friendly in parallel streams?
      → Hint: stateless transformation

192. Why should mutable reductions be avoided in streams?
      → Hint: race problems

193. Why is forEach not recommended for parallel streams?
      → Hint: order chaos

194. How does a StreamBuilder work?
      → Hint: collects elements before building

195. What makes EnumMap memory-efficient?
      → Hint: uses ordinal as index

196. Why is NavigableSet useful?
      → Hint: nearest neighbours

197. How does Objects.compare use a comparator?
      → Hint: fallback to rules

198. Why is Collections.rotate an efficient operation?
      → Hint: uses reversal trick

199. What is the difference between toArray() and toArray(T[])?
      → Hint: type-safety vs cast

200. Why is Arrays.sort faster for primitives?
      → Hint: tuned algorithms

---

### ✔️ **Batch 2 completed!**

---

# ✅ **Batch 3 — Advanced Concurrency & Multithreading (Q201–Q300)**

### **One-Line Hard Questions + Tiny Hints**

201. What happens when two threads try to update the same variable at the same time?
      → Hint: race to chaos

202. Why is `synchronized` not always enough for thread safety?
      → Hint: slow + not flexible

203. How does intrinsic locking actually work?
      → Hint: every object carries a tiny lock

204. Why is lock contention harmful?
      → Hint: threads waiting in long lines

205. What is a monitor in Java?
      → Hint: lock + waiting room

206. Why can't you lock on a `String` literal?
      → Hint: shared across the whole app

207. What is reentrant locking?
      → Hint: same thread allowed back in

208. Why does Java use biased locking?
      → Hint: speed optimisation for single-threaded use

209. What is lock inflation?
      → Hint: small lock → big lock when needed

210. Why are volatile variables not atomic?
      → Hint: visibility ≠ protection

211. How does a volatile write create a memory barrier?
      → Hint: stops reordering

212. What is false sharing?
      → Hint: cache line fighting

213. Why is CAS (compare-and-swap) important?
      → Hint: lock-free decision making

214. What is the ABA problem?
      → Hint: looks same but changed

215. How do Atomic classes solve the ABA problem?
      → Hint: version stamps or careful loops

216. Why does AtomicInteger use a CAS loop?
      → Hint: retry until success

217. What is a happens-before relationship?
      → Hint: ordering rules for memory

218. Why is the Java Memory Model needed?
      → Hint: CPUs reorder stuff

219. What is the difference between a thread and a task?
      → Hint: worker vs job

220. Why is creating many threads expensive?
      → Hint: memory + context switching

221. How does a thread pool improve performance?
      → Hint: reuse workers

222. What is the difference between fixed and cached thread pools?
      → Hint: steady vs elastic

223. Why is ForkJoinPool good for divide-and-conquer tasks?
      → Hint: steal unused work

224. What is work-stealing?
      → Hint: idle worker grabs tasks from others

225. Why does ExecutorService need shutdown()?
      → Hint: threads keep running otherwise

226. What is the difference between shutdown() and shutdownNow()?
      → Hint: polite vs rude

227. Why does Future.get block?
      → Hint: waits for result

228. What happens if you cancel a Future?
      → Hint: may interrupt thread

229. What is the purpose of Callable vs Runnable?
      → Hint: return value + exceptions

230. Why is CompletableFuture so powerful?
      → Hint: chaining + async magic

231. How does CompletableFuture support non-blocking execution?
      → Hint: callback style

232. What is a thread starvation problem?
      → Hint: some never get CPU time

233. Why is using synchronized + wait/notify tricky?
      → Hint: easy to make mistakes

234. What is spurious wakeup?
      → Hint: wake without reason

235. Why must wait() always be inside a loop?
      → Hint: recheck condition

236. What is a CountDownLatch used for?
      → Hint: wait for group to finish

237. Why can't a CountDownLatch be reused?
      → Hint: hits zero once

238. What is a CyclicBarrier used for?
      → Hint: threads meet at checkpoints

239. Why is CyclicBarrier reusable?
      → Hint: reset after meeting

240. What is a Semaphore used for?
      → Hint: limit number of entries

241. What is a fair semaphore?
      → Hint: first-come-first-served

242. What is a ReentrantLock used for?
      → Hint: flexible alternative to synchronized

243. Why use tryLock()?
      → Hint: avoid waiting forever

244. What is a read-write lock?
      → Hint: many readers, one writer

245. Why should you avoid holding locks during I/O?
      → Hint: long blocking

246. Why is double-checked locking used?
      → Hint: reduce locking cost

247. Why was double-checked locking broken before Java 5?
      → Hint: memory model issues

248. What is an executor rejection policy?
      → Hint: what to do with extra tasks

249. How does ThreadLocal maintain data?
      → Hint: per-thread mini storage

250. Why can ThreadLocal cause memory leaks?
      → Hint: thread pools never die

251. What is a daemon thread?
      → Hint: helper that doesn’t block JVM exit

252. Why should you avoid long operations in finalizers?
      → Hint: GC delays

253. What is a thread dump?
      → Hint: snapshot of all threads

254. What is a livelock?
      → Hint: dancing but not progressing

255. What is a deadlock?
      → Hint: threads blocking each other forever

256. Why are circular waits dangerous?
      → Hint: deadlock ingredient

257. How does timeout locking reduce deadlock chances?
      → Hint: gives up eventually

258. What is a memory fence?
      → Hint: stop-reorder wall

259. What is the difference between stack and heap for threads?
      → Hint: each has stack, share heap

260. Why are thread priorities unreliable?
      → Hint: OS decides real scheduling

261. What is fork/join recursive decomposition?
      → Hint: split tasks into smaller tasks

262. Why must fork/join tasks be small?
      → Hint: overhead grows fast

263. What is a Phaser used for?
      → Hint: advanced barrier

264. Why is interrupting a thread cooperative?
      → Hint: thread must check flag

265. How does Thread.interrupted differ from isInterrupted()?
      → Hint: one clears flag

266. Why should interrupts be handled promptly?
      → Hint: responsive execution

267. Why is busy-waiting bad?
      → Hint: CPU spins for nothing

268. What is a spinlock?
      → Hint: waiting by spinning

269. Why might spinning be faster sometimes?
      → Hint: short waits only

270. Why are concurrent collections better than synchronized wrappers?
      → Hint: fine-grained locking

271. What is LongAdder?
      → Hint: counters without contention

272. Why is LongAdder faster than AtomicLong under heavy load?
      → Hint: striped counters

273. How does Phaser handle dynamic thread registration?
      → Hint: threads can join/leave

274. What is barrierAction in CyclicBarrier?
      → Hint: action after all arrive

275. Why avoid nested locks?
      → Hint: deadlock trap

276. Why are futures not great for complex async flows?
      → Hint: no chaining

277. What is a bounded queue in executors for?
      → Hint: control overload

278. Why is SynchronousQueue special?
      → Hint: zero capacity

279. What is a thread-safe lazy initialization technique?
      → Hint: holder pattern

280. Why is Idle CPU time important in concurrency?
      → Hint: prevents overheating + context thrash

281. What is a striped lock?
      → Hint: many small locks instead of one big

282. Why use a countdown event before starting threads?
      → Hint: synchronized start

283. What is task stealing in ForkJoin?
      → Hint: idle thread helps peers

284. Why avoid blocking calls in parallel streams?
      → Hint: ruins parallelism

285. What is a non-blocking algorithm?
      → Hint: no locks, retry loops

286. Why does CAS rely on hardware?
      → Hint: atomic primitives

287. What is a memory visibility bug?
      → Hint: updated value not seen

288. Why must shared variables be guarded?
      → Hint: avoid corruption

289. What is a happens-before edge created by lock release?
      → Hint: write-then-read guarantee

290. Why should thread pools be sized based on CPUs?
      → Hint: optimal use

291. What is fork-join "async mode"?
      → Hint: pushing tasks externally

292. Why is fairness optional in locks?
      → Hint: fairness slows things down

293. Why is StampedLock faster than ReentrantReadWriteLock?
      → Hint: optimistic reads

294. How does optimistic read work?
      → Hint: check stamp at end

295. Why is unsafe used internally in concurrency utilities?
      → Hint: low-level power

296. Why should you not write your own lock algorithms?
      → Hint: tricky and unsafe

297. Why is ExecutorCompletionService useful?
      → Hint: get tasks as they complete

298. Why are blocking queues essential for producer-consumer?
      → Hint: automatic waiting/waking

299. What is the difference between fair and unfair locks?
      → Hint: queue order vs speed

300. Why should concurrent code prefer immutability?
      → Hint: safe by default

---

### ✔️ **Batch 3 completed!**

---

# ✅ **Batch 4 — Java I/O, NIO, Serialization & Networking (Q301–Q400)**

### **One-Line Hard Questions + Tiny Hints**

301. Why are streams in Java unidirectional?
      → Hint: in or out, not both

302. What is the difference between byte streams and character streams?
      → Hint: raw data vs text-friendly

303. Why does Reader/Writer use Unicode?
      → Hint: world languages club

304. What is buffering in I/O?
      → Hint: take a bucket, not a spoon

305. Why is BufferedReader faster than FileReader?
      → Hint: reduces disk trips

306. What makes FileInputStream slower compared to BufferedInputStream?
      → Hint: tiny reads cost more

307. What problem does Try-with-resources solve?
      → Hint: no more forgetting to close

308. Why does File class not represent actual files only?
      → Hint: path info, not file data

309. What is the difference between File and Path (NIO)?
      → Hint: modern vs old school

310. Why does NIO use Channels instead of Streams?
      → Hint: bidirectional + faster

311. What is a Buffer in NIO?
      → Hint: container with position limits

312. How does a Channel differ from a Stream?
      → Hint: both-way street

313. Why is NIO called non-blocking?
      → Hint: thread doesn't sit idle

314. What is the purpose of Selectors?
      → Hint: watch many channels at once

315. Why does ByteBuffer have direct and non-direct types?
      → Hint: memory location differences

316. Why are direct buffers faster?
      → Hint: skip JVM heap copy

317. Why are direct buffers slower to create?
      → Hint: OS-level allocation

318. What is scattering read?
      → Hint: read → many buffers

319. What is gathering write?
      → Hint: write ← many buffers

320. How do you flip a buffer?
      → Hint: read mode switch

321. What is the difference between position and limit?
      → Hint: where you are vs where you stop

322. Why is clear() not clearing data?
      → Hint: resets pointers, not bytes

323. What is memory-mapped file I/O?
      → Hint: file meets RAM directly

324. Why are memory-mapped files great for large files?
      → Hint: OS paging magic

325. What’s the risk with memory-mapped files?
      → Hint: out-of-memory surprises

326. What is a FileChannel used for?
      → Hint: efficient file access

327. Why does ObjectInputStream require serialVersionUID?
      → Hint: version control

328. What happens if serialVersionUID mismatches?
      → Hint: boom — InvalidClassException

329. Why is serialization dangerous?
      → Hint: hidden code execution risks

330. Why is transient keyword used?
      → Hint: skip sensitive fields

331. What is Externalizable used for?
      → Hint: full control over serialization

332. Why must Externalizable have a public no-arg constructor?
      → Hint: object needs to be recreated

333. Why is default Java serialization slow?
      → Hint: reflection + metadata

334. Why does serialization break encapsulation?
      → Hint: private fields exposed

335. Why should you avoid serializing large object graphs?
      → Hint: deep and heavy

336. What is object graph traversal in serialization?
      → Hint: crawler visits all connected objects

337. Why is writeReplace useful?
      → Hint: swap objects before serialization

338. Why is readResolve used?
      → Hint: enforce singletons

339. How does serialization handle cyclic references?
      → Hint: keeps a reference table

340. What is socket programming used for?
      → Hint: chat between computers

341. Difference between TCP and UDP?
      → Hint: reliable vs speedy

342. What is a port in networking?
      → Hint: mailbox number

343. Why is TCP connection-oriented?
      → Hint: handshake first

344. What is the purpose of the 3-way handshake?
      → Hint: “you there?” → “yes!”*

345. What is Nagle’s algorithm?
      → Hint: bundle tiny packets

346. Why is TCP slower than UDP?
      → Hint: reliability comes at cost

347. What is a datagram?
      → Hint: tiny self-contained packet

348. Why use DatagramSocket?
      → Hint: lightweight communication

349. What is a ServerSocket?
      → Hint: gatekeeper for clients

350. Why does accept() block?
      → Hint: waits for a caller

351. How do you make a socket timeout?
      → Hint: setSoTimeout

352. Why use socket keepalive?
      → Hint: check if the peer is alive

353. Why is buffering important in networking?
      → Hint: control packet sizes

354. What is the purpose of InputStream vs BufferedInputStream in networking sockets?
      → Hint: raw vs buffered data

355. Why is non-blocking NIO good for many connections?
      → Hint: one thread watches all

356. What is the difference between blocking and non-blocking socket modes?
      → Hint: wait vs don’t wait

357. Why does Selector work with channels only?
      → Hint: streams can't register

358. How does a SelectionKey work?
      → Hint: tells what event happened

359. Why are interest ops (OP_READ, OP_WRITE) important?
      → Hint: monitor specific events

360. Why is networking I/O usually slower than disk I/O?
      → Hint: long travel distance

361. How does URLConnection handle redirects?
      → Hint: follows rules

362. Why is HttpURLConnection old and awkward?
      → Hint: early design

363. Why is the Java HTTP Client (Java 11+) better?
      → Hint: async + simpler

364. How does chunked transfer work?
      → Hint: send pieces without knowing final size

365. Why does buffering reduce number of system calls?
      → Hint: fewer trips

366. Why are file descriptors limited?
      → Hint: OS resource caps

367. What is zero-copy I/O?
      → Hint: skip copying between buffers

368. How does `transferTo()` implement zero-copy?
      → Hint: OS sends file directly

369. Why is file locking needed?
      → Hint: avoid two writers clashing

370. What is the difference between shared and exclusive file locks?
      → Hint: read together, write alone

371. Why are file locks across processes tricky?
      → Hint: OS differences

372. Why should channels be closed explicitly?
      → Hint: free file handles

373. What is Path normalization?
      → Hint: remove ../ and ./

374. Why does Files.walk potentially cause memory issues?
      → Hint: huge directory trees

375. What is WatchService?
      → Hint: file system change watcher

376. Why does WatchService sometimes miss events?
      → Hint: platform limits

377. Why is asynchronous file I/O useful?
      → Hint: thread doesn’t block

378. What is AsynchronousChannelGroup?
      → Hint: shared thread pool

379. Why does FileStore not always report correct values?
      → Hint: OS abstraction quirks

380. Why is RandomAccessFile special?
      → Hint: jump anywhere

381. Difference between seek() and skip()?
      → Hint: precise move vs relative hop

382. Why is DataInputStream useful?
      → Hint: read primitives easily

383. Why are object streams sensitive to class changes?
      → Hint: stored metadata

384. Why are socket streams buffered by default?
      → Hint: efficiency trick

385. What is half-close in TCP?
      → Hint: close write but keep read

386. Why is socket linger dangerous?
      → Hint: blocks close

387. What is a DNS lookup?
      → Hint: find server address

388. Why does DNS caching matter?
      → Hint: speed boost

389. Why is hostname resolution slow sometimes?
      → Hint: remote server lookup

390. What is ICMP unreachable?
      → Hint: target can’t be reached

391. How does proxying work in HTTP?
      → Hint: middleman server

392. Why is SSL/TLS handshake expensive?
      → Hint: cryptography setup

393. What is certificate validation?
      → Hint: is the site trustworthy?*

394. Why must InputStreams be read until -1?
      → Hint: -1 means “end”

395. Why are file permissions OS-dependent?
      → Hint: Java just reports them

396. What is EOFException?
      → Hint: no more data but still asked

397. Why is streaming large files through sockets slow?
      → Hint: network bottleneck

398. Why does NIO perform better with selectors than with threads?
      → Hint: fewer threads waiting

399. What is a stale file handle?
      → Hint: file vanished but handle still exists

400. Why are directories sometimes treated like files in low-level I/O?
      → Hint: OS treats everything as files

---

### ✔️ **Batch 4 completed!**

---

# ✅ **Batch 5 — JDBC, ORM, JPA & Hibernate (Q401–Q500)**

### **One-Line Hard Questions + Tiny Hints**

401. Why must JDBC connections always be closed?
      → Hint: they’re like borrowed bikes

402. What is the difference between Statement and PreparedStatement?
      → Hint: plain vs pre-planned

403. Why are PreparedStatements safer?
      → Hint: SQL injection shield

404. Why do PreparedStatements improve performance?
      → Hint: server caches the plan

405. What is a ResultSet cursor?
      → Hint: tiny pointer on a table

406. Why do some ResultSets become forward-only?
      → Hint: driver limits

407. Why do JDBC drivers vary in features?
      → Hint: vendor choices

408. What is connection pooling?
      → Hint: reuse connections like shared taxis

409. Why is connection pooling essential in production?
      → Hint: opening connections is slow

410. What is auto-commit mode?
      → Hint: every query = instant commit

411. Why should auto-commit often be disabled?
      → Hint: control your transactions

412. What does rollback actually do?
      → Hint: undo changes before commit

413. Why do long transactions harm performance?
      → Hint: they block others

414. What is batch processing in JDBC?
      → Hint: send many queries at once

415. Why are batches faster?
      → Hint: fewer round trips

416. What is the difference between JDBC driver types (1–4)?
      → Hint: old to newest styles

417. Why is the type 4 driver preferred?
      → Hint: pure Java, no middleman

418. What is metadata in JDBC?
      → Hint: info about your tables

419. What is DatabaseMetaData used for?
      → Hint: learn about DB structure

420. What is ResultSetMetaData used for?
      → Hint: learn about query columns

421. Why are savepoints helpful?
      → Hint: mini checkpoints

422. Why should you avoid SELECT *?
      → Hint: too much baggage

423. How does ORM remove boilerplate?
      → Hint: maps objects to tables automatically

424. What problem does ORM try to solve?
      → Hint: mismatch between objects and tables

425. What is the object-relational impedance mismatch?
      → Hint: objects ≠ tables

426. What is a JPA Entity?
      → Hint: a class that maps to a DB table

427. Why must entities have a no-arg constructor?
      → Hint: frameworks need to build them

428. What is @Id used for?
      → Hint: choose the key field

429. Why should primary keys never change?
      → Hint: identity must stay fixed

430. What is GenerationType.IDENTITY?
      → Hint: DB decides the ID

431. What is GenerationType.SEQUENCE?
      → Hint: database counter

432. Why use GenerationType.TABLE?
      → Hint: universal fallback

433. Why does @Column allow nullable, length, unique?
      → Hint: fine-tune field rules

434. What is an entity lifecycle?
      → Hint: new → managed → detached → removed

435. What is the managed state?
      → Hint: entity under Hibernate’s watch

436. Why are detached entities tricky?
      → Hint: not tracked anymore

437. What is merge() used for?
      → Hint: reconnect lost entities

438. Why is persist() different from merge()?
      → Hint: persist = new child; merge = lost child returns

439. What is the difference between flush() and commit()?
      → Hint: send changes vs finish transaction

440. Why doesn’t flush() save to the database permanently?
      → Hint: commit decides the fate

441. What is the persistence context?
      → Hint: entity storage room

442. Why does JPA use dirty checking?
      → Hint: detect changes automatically

443. What is lazy loading?
      → Hint: fetch data only when needed

444. Why is lazy loading dangerous outside a transaction?
      → Hint: LazyInitializationException party

445. What is eager loading?
      → Hint: load everything upfront

446. Why is eager loading risky?
      → Hint: too much data too soon

447. How do you map a one-to-one relation?
      → Hint: @OneToOne

448. How do you map a one-to-many relation?
      → Hint: @OneToMany

449. Why is the owning side important in relationships?
      → Hint: it writes the foreign key

450. What is mappedBy used for?
      → Hint: specify the non-owning side

451. Why do bidirectional mappings cause infinite loops in JSON?
      → Hint: parent → child → parent → child…*

452. Why use DTOs instead of entities for APIs?
      → Hint: keep things clean and safe

453. What is cascading?
      → Hint: operations flow to child entities

454. Why is CascadeType.ALL dangerous?
      → Hint: deletes may spread everywhere

455. What is orphanRemoval?
      → Hint: delete lonely children

456. Why is @Embeddable useful?
      → Hint: reusable value objects

457. What are entity listeners?
      → Hint: lifecycle event hooks

458. Why use @PrePersist or @PreUpdate?
      → Hint: auto-set timestamps

459. What is JPQL?
      → Hint: SQL for entity objects

460. Why can't JPQL query tables?
      → Hint: it works with entity models

461. What is Criteria API?
      → Hint: type-safe queries

462. Why is Criteria API verbose?
      → Hint: lots of objects to build queries

463. Why is native SQL sometimes required?
      → Hint: complex or vendor-specific queries

464. What is the N+1 select problem?
      → Hint: one main query + many tiny queries

465. Why is fetch join used?
      → Hint: kill N+1 problem

466. Why should fetch join be used carefully?
      → Hint: can explode result size

467. What is second-level cache?
      → Hint: app-wide memory for entities

468. Why does Hibernate offer query cache separately?
      → Hint: store results, not entities

469. Why can caching cause stale data?
      → Hint: cache isn't always aware of DB changes

470. What is a natural key vs surrogate key?
      → Hint: business value vs generated value

471. Why prefer surrogate keys?
      → Hint: stable and simple

472. What is optimistic locking?
      → Hint: assume no conflict; verify with version

473. Why does optimistic locking use @Version?
      → Hint: check if data changed

474. What is pessimistic locking?
      → Hint: lock first, then update

475. Why avoid pessimistic locking?
      → Hint: blocks other users

476. Why does Hibernate batch inserts?
      → Hint: fewer DB trips = faster

477. What is flush mode?
      → Hint: controls when changes are synced

478. Why is FlushMode.COMMIT faster?
      → Hint: no mid-transaction updates

479. What is @Transactional used for?
      → Hint: define boundaries

480. Why must @Transactional be on public methods?
      → Hint: proxies wrap them

481. Why doesn’t @Transactional work on internal method calls?
      → Hint: self-calls skip proxies

482. Why should transactions be short?
      → Hint: frees locks sooner

483. Why is isolation level important?
      → Hint: controls read/write interference

484. What is dirty read?
      → Hint: reading uncommitted data

485. What is non-repeatable read?
      → Hint: value changes mid-read

486. What is phantom read?
      → Hint: new rows appear

487. Why is READ_COMMITTED default in many databases?
      → Hint: good balance

488. What is ACID in transactions?
      → Hint: safety rules

489. Why does ORM struggle with bulk updates?
      → Hint: bypasses entity lifecycle

490. Why are bulk operations dangerous in ORM?
      → Hint: persistence context becomes outdated

491. Why prefer native SQL for large batch updates?
      → Hint: ORM too slow for huge sets

492. Why is flush() needed before bulk queries?
      → Hint: avoid stale writes

493. Why is clear() sometimes needed after batch writes?
      → Hint: free memory + avoid inconsistencies

494. Why does Hibernate sometimes trigger SELECT before UPDATE?
      → Hint: dirty-checking needs comparison

495. How does Hibernate detect changes in entities?
      → Hint: snapshot comparison

496. Why does join fetching override lazy loading?
      → Hint: fetch join = must load

497. Why does Hibernate sometimes reorder SQL statements?
      → Hint: foreign key rules

498. What is the difference between session and session factory?
      → Hint: one is a machine, one is a product

499. Why is SessionFactory thread-safe?
      → Hint: shared by all

500. Why must Session not be shared across threads?
      → Hint: not thread-safe

---

### ✔️ **Batch 5 completed!**

---

# ✅ **Batch 6 — Spring Core, DI, IoC & AOP (Q501–Q600)**

### **One-Line Hard Questions + Tiny Hints**

501. What problem does IoC fundamentally solve?
      → Hint: objects stop babysitting their dependencies

502. Why is dependency injection preferred over manual object creation?
      → Hint: plug-and-play instead of DIY

503. What is the difference between IoC and DI?
      → Hint: DI is one way to achieve IoC

504. Why is the Spring container called a “container”?
      → Hint: holds and manages objects like a warehouse

505. What is a Spring Bean?
      → Hint: an object Spring babysits

506. Why must beans have predictable constructors?
      → Hint: container needs to create them easily

507. What is a BeanFactory?
      → Hint: basic bean maker

508. How does ApplicationContext differ from BeanFactory?
      → Hint: BeanFactory on steroids

509. Why does ApplicationContext eagerly initialize beans?
      → Hint: fewer surprises later

510. What is lazy initialization of beans?
      → Hint: make when needed

511. Why is @Component used?
      → Hint: tell Spring “this class is yours”

512. Why are @Service and @Repository just stereotypes?
      → Hint: same thing with better names

513. What is classpath scanning?
      → Hint: Spring hunts for beans

514. Why use @Configuration?
      → Hint: tells Spring “bean recipes here”

515. What is @Bean used for?
      → Hint: create bean manually

516. Why is @Bean method preferred for external libraries?
      → Hint: can’t annotate third-party classes

517. Why is constructor injection recommended?
      → Hint: avoids nulls and surprises

518. Why avoid field injection?
      → Hint: testing becomes painful

519. Why does setter injection exist?
      → Hint: optional dependencies

520. What is circular dependency in Spring?
      → Hint: A needs B, B needs A

521. Why does Spring fail on constructor-based circular dependencies?
      → Hint: can’t create either one first

522. Why does Spring allow circular dependencies in setters?
      → Hint: inject after creation

523. Why is circular dependency a bad design smell?
      → Hint: too much mutual hugging

524. What is bean scope?
      → Hint: how long a bean lives

525. Why is singleton the default scope?
      → Hint: memory-friendly and stable

526. What is prototype scope for?
      → Hint: fresh object every time

527. Why avoid prototype beans in singleton beans?
      → Hint: stale instances

528. What is request scope?
      → Hint: one bean per web request

529. What is session scope?
      → Hint: one bean per user session

530. Why can’t prototype beans be auto-destroyed?
      → Hint: container doesn't track them

531. What is the bean lifecycle?
      → Hint: create → populate → init → use → destroy

532. Why use @PostConstruct?
      → Hint: run code after setup

533. Why use @PreDestroy?
      → Hint: cleanup duty

534. What are BeanPostProcessors?
      → Hint: inspectors that modify beans

535. What is the importance of the BeanFactoryPostProcessor?
      → Hint: tweak bean definitions before creation

536. Why is PropertySourcesPlaceholderConfigurer used?
      → Hint: resolves ${variables}

537. What is Environment abstraction?
      → Hint: access properties safely

538. What is the difference between @Value and @ConfigurationProperties?
      → Hint: single value vs full mapping

539. Why is @ConfigurationProperties preferred?
      → Hint: cleaner and type-safe

540. What is the purpose of @Profile?
      → Hint: load beans conditionally

541. Why use profiles in production?
      → Hint: dev/stage/prod differences

542. What is bean autowiring?
      → Hint: automatic matching

543. Why might autowiring fail?
      → Hint: more than one candidate

544. Why use @Qualifier?
      → Hint: pick the right bean

545. Why is component scanning sometimes dangerous?
      → Hint: accidental beans everywhere

546. Why is explicit configuration more predictable?
      → Hint: you know what you get

547. What is AOP in Spring?
      → Hint: add extra behaviour without touching code

548. What problem does AOP solve?
      → Hint: remove cross-cutting clutter

549. What is a join point?
      → Hint: “where” advice can happen

550. What is a pointcut?
      → Hint: rule to choose join points

551. What is advice?
      → Hint: extra code you inject

552. What are the types of advice?
      → Hint: before, after, around

553. Why is around advice the most powerful?
      → Hint: you can run + skip + wrap

554. What is a proxy in Spring AOP?
      → Hint: a wrapper around your object

555. Why does Spring AOP use proxies instead of bytecode weaving?
      → Hint: simpler + runtime

556. What is JDK dynamic proxy?
      → Hint: proxy for interfaces

557. When does Spring use CGLIB proxy?
      → Hint: proxy for classes

558. Why does AOP not work on private methods?
      → Hint: proxy can’t see them

559. Why does AOP not run on self-invocation?
      → Hint: calling yourself bypasses proxy

560. What is the concept of cross-cutting concerns?
      → Hint: logic used everywhere

561. Why is logging a cross-cutting concern?
      → Hint: appears in all layers

562. Why does AOP help reduce boilerplate?
      → Hint: no repeated code

563. What is advice ordering?
      → Hint: which advice runs first

564. Why is @Order used?
      → Hint: tidy sequence of advice

565. Why must aspects be Spring beans?
      → Hint: container manages them

566. What is JoinPoint.getArgs() used for?
      → Hint: inspect method inputs

567. What is ProceedingJoinPoint?
      → Hint: lets around advice continue

568. Why avoid heavy logic in aspects?
      → Hint: they run everywhere

569. Why can aspects hurt performance?
      → Hint: proxy overhead

570. What is aspect instantiation model?
      → Hint: when aspect instances are created

571. What is @EnableAspectJAutoProxy for?
      → Hint: turn AOP switch on

572. Why doesn't AOP work without this switch?
      → Hint: proxies aren’t created

573. What is the difference between compile-time and runtime weaving?
      → Hint: build phase vs running phase

574. Why is Spring AOP limited to method-level weaving?
      → Hint: proxies only handle methods

575. Why not use AOP for business logic?
      → Hint: it’s for supportive tasks only

576. Why do aspects sometimes cause circular dependencies?
      → Hint: proxies wrapping proxies

577. Why use annotation-based AOP?
      → Hint: simple and readable

578. Why does @Transactional rely on AOP?
      → Hint: adds transactional behaviour

579. Why doesn’t @Transactional work on final classes?
      → Hint: CGLIB can’t subclass them

580. Why avoid final methods with AOP?
      → Hint: cannot intercept

581. What is bean overriding?
      → Hint: replace one bean with another

582. Why is bean overriding often disabled?
      → Hint: prevents surprise replacements

583. What is ApplicationEventPublisher?
      → Hint: send events

584. Why use application events?
      → Hint: decoupled communication

585. What is @EventListener?
      → Hint: catch events easily

586. What is the purpose of ContextRefreshedEvent?
      → Hint: container loaded

587. Why does Spring separate core and context modules?
      → Hint: modularity

588. What is Resource abstraction?
      → Hint: unified access to files, URLs, classpath

589. Why use ClassPathResource?
      → Hint: get files inside JAR

590. Why use @Lazy?
      → Hint: load bean only when needed

591. Why does Spring prefer immutability?
      → Hint: safer and predictable

592. What is @Primary used for?
      → Hint: default candidate

593. Why do bean names matter?
      → Hint: resolve conflicts

594. Why is ApplicationContext closed explicitly sometimes?
      → Hint: release resources

595. What is BeanDefinition?
      → Hint: blueprint for beans

596. Why is bean wiring called “automatic”?
      → Hint: container finds dependencies

597. What does @DependsOn guarantee?
      → Hint: initialization order

598. Why use property files with Spring?
      → Hint: external configuration

599. What is the purpose of the Spring Expression Language?
      → Hint: mini calculator inside Spring

600. Why avoid overly complex SpEL expressions?
      → Hint: unreadable and error-prone

---

### ✔️ Batch 6 complete!

---

# ✅ **Batch 7 — Spring Boot & REST (Q601–Q700)**

### **One-Line Hard Questions + Tiny Hints**

601. What problem does Spring Boot mainly solve?
      → Hint: stop wiring everything manually

602. Why are starters used in Spring Boot?
      → Hint: bundles of ready-made stuff

603. Why does Spring Boot rely on auto-configuration?
      → Hint: scans → guesses → configures

604. Why must auto-configuration classes run last?
      → Hint: let your configs win

605. What is the purpose of @SpringBootApplication?
      → Hint: three annotations packed into one

606. Why should main class be in the root package?
      → Hint: component scanning works outward

607. How does Spring Boot detect missing beans?
      → Hint: conditional magic

608. What is @ConditionalOnMissingBean used for?
      → Hint: only create if not present

609. What is @ConditionalOnProperty used for?
      → Hint: toggle features

610. Why use externalized configuration?
      → Hint: change settings without touching code

611. What is application.properties?
      → Hint: Boot’s favourite notebook

612. Why support application.yml?
      → Hint: cleaner structure

613. Why use @ConfigurationProperties?
      → Hint: map whole config sections

614. Why does Boot support multiple property sources?
      → Hint: override rules

615. What is the order of config loading in Boot?
      → Hint: file → env → args

616. Why is actuator useful?
      → Hint: health check superhero

617. Why protect actuator endpoints?
      → Hint: too much info to be public

618. What is embedded Tomcat for?
      → Hint: app carries its own server

619. Why does embedded server simplify deployments?
      → Hint: one jar to rule them all

620. What is auto-configuration report?
      → Hint: who configured what

621. Why disable specific auto-configurations sometimes?
      → Hint: avoid conflicts

622. Why are @RestController and @Controller different?
      → Hint: REST auto-adds @ResponseBody

623. Why is @ResponseBody needed?
      → Hint: output goes to JSON, not view

624. Why use @GetMapping etc. instead of @RequestMapping?
      → Hint: clearer and cleaner

625. What is the importance of @PathVariable?
      → Hint: extract ID from URL

626. Why use @RequestParam?
      → Hint: query parameters helper

627. What is @RequestBody used for?
      → Hint: read JSON payload

628. Why is validation needed in REST?
      → Hint: protect your API from weird inputs

629. What is @Valid for?
      → Hint: check incoming data

630. Why does REST need proper HTTP status codes?
      → Hint: universal language

631. Why use ResponseEntity?
      → Hint: full control of response

632. Why do APIs need versioning?
      → Hint: new changes shouldn’t break old users

633. Why is REST stateless?
      → Hint: each request stands alone

634. Why is idempotency important?
      → Hint: same call, same effect

635. Why is PUT idempotent but POST not?
      → Hint: replace vs create

636. Why are REST APIs vulnerable to over-fetching?
      → Hint: too much data sent

637. Why are they vulnerable to under-fetching?
      → Hint: missing needed data

638. Why use DTO instead of entity for APIs?
      → Hint: avoid exposing secrets

639. Why do we need exception handlers?
      → Hint: clean error messages

640. What is @ControllerAdvice?
      → Hint: central error hub

641. Why do we use custom error responses?
      → Hint: humans understand better

642. Why enable CORS?
      → Hint: browsers are strict guardians

643. Why restrict CORS properly?
      → Hint: safety first

644. What is HATEOAS?
      → Hint: links inside responses

645. Why is HATEOAS rarely used?
      → Hint: complex for most apps

646. Why use content negotiation?
      → Hint: JSON or XML? Your choice

647. Why define produces/consumes attributes?
      → Hint: control formats

648. Why use interceptors in REST?
      → Hint: pre/post processing

649. Why prefer filters for request-wide logic?
      → Hint: runs on every request

650. Why are filters lower level than interceptors?
      → Hint: servlet world vs Spring world

651. What is the DispatcherServlet?
      → Hint: the traffic police of Spring MVC

652. Why is HandlerMapping important?
      → Hint: decide which controller handles what

653. Why disable default error page in Boot?
      → Hint: show custom JSON errors

654. Why is async request handling useful?
      → Hint: free threads sooner

655. What is DeferredResult used for?
      → Hint: respond later

656. Why use WebClient instead of RestTemplate?
      → Hint: async + reactive

657. Why is RestTemplate deprecated-ish?
      → Hint: old style, no async superpowers

658. Why does WebClient use reactive streams?
      → Hint: handles many requests efficiently

659. Why does Boot support multiple JSON libraries?
      → Hint: flexibility

660. Why is ObjectMapper expensive to create?
      → Hint: lots of configuration

661. Why re-use a single ObjectMapper bean?
      → Hint: save performance

662. Why use @JsonIgnore?
      → Hint: hide unwanted fields

663. Why is circular reference a problem in JSON?
      → Hint: infinite loops

664. Why use Spring Cache abstraction?
      → Hint: speed up repeated calls

665. Why is caching dangerous in REST?
      → Hint: stale data

666. What is key generation in caching?
      → Hint: how results are remembered

667. Why use async methods (@Async)?
      → Hint: background tasks

668. Why must @Async work on separate beans?
      → Hint: proxy magic again

669. Why does Boot provide metrics?
      → Hint: see app health and performance

670. Why expose health checks externally?
      → Hint: load balancers need info

671. Why secure health endpoints in production?
      → Hint: avoid exposing secrets

672. What is graceful shutdown?
      → Hint: finish work before stopping

673. Why use @EnableWebMvc carefully?
      → Hint: overrides Boot defaults

674. Why is server compression used?
      → Hint: save bandwidth

675. Why avoid large response bodies?
      → Hint: slow clients, high memory

676. Why rate-limit JSON parsing?
      → Hint: protection from huge payloads

677. Why do REST APIs need pagination?
      → Hint: handle big data safely

678. Why use Sort and Pageable interfaces?
      → Hint: Spring-data magic

679. Why use @RepositoryRestResource?
      → Hint: auto-generate REST endpoints

680. Why is it dangerous in production?
      → Hint: too much is exposed

681. Why use HAL browser?
      → Hint: visualize API links

682. Why use Sleuth with REST?
      → Hint: trace requests across services

683. Why is correlation ID important?
      → Hint: find one request in logs

684. Why secure REST with JWT?
      → Hint: no server-side sessions

685. Why is JWT stateless?
      → Hint: token carries info

686. Why can JWT be dangerous?
      → Hint: stolen token = game over

687. Why set JWT expiration times?
      → Hint: safety timer

688. Why refresh tokens exist?
      → Hint: extend login without re-login

689. Why is CSRF not required for JWT APIs?
      → Hint: no cookies usually

690. Why use API keys?
      → Hint: simple authentication

691. Why avoid storing API keys in code?
      → Hint: secrets leak

692. Why use rate limiting?
      → Hint: keep API alive

693. Why enable gzip in server?
      → Hint: faster client experience

694. Why return ETags?
      → Hint: caching validation

695. Why use If-Modified-Since?
      → Hint: avoid sending large unchanged data

696. Why log request/response bodies carefully?
      → Hint: sensitive information

697. Why prefer idempotent endpoints for retries?
      → Hint: safe repeat

698. Why validate path variables?
      → Hint: never trust input

699. Why monitor REST latency?
      → Hint: detect slow endpoints

700. Why avoid blocking calls inside REST endpoints?
      → Hint: blocks threads → slow server

---

### ✔️ Batch 7 complete!

---

# ✅ **Batch 8 — Microservices, Spring Cloud & Distributed Systems (Q701–Q800)**

### **One-Line Hard Questions + Tiny Hints**

701. Why do microservices exist?
      → Hint: break the giant into friendly bite-sized pieces

702. Why must microservices be independently deployable?
      → Hint: each kid should get ready without waiting for siblings

703. Why do microservices need strong boundaries?
      → Hint: no stepping into each other’s rooms

704. Why is loose coupling essential?
      → Hint: less drama between services

705. Why is high cohesion needed?
      → Hint: keep related stuff together

706. Why does each microservice need its own database?
      → Hint: no shared notebooks to fight over

707. Why is synchronous communication risky?
      → Hint: one slow friend delays everyone

708. Why use asynchronous messaging?
      → Hint: send and forget, like an email

709. Why do microservices need an API gateway?
      → Hint: one door for all services

710. Why should APIs behind gateways stay hidden?
      → Hint: keep internal rooms private

711. Why use load balancing?
      → Hint: share the workload nicely

712. Why do we need service discovery?
      → Hint: services must find each other like GPS

713. What does Eureka provide?
      → Hint: phonebook of services

714. Why are heartbeats important in Eureka?
      → Hint: check who’s still alive

715. What is Ribbon used for?
      → Hint: client-side load balancing

716. Why is Feign easier than RestTemplate?
      → Hint: write interfaces, not code

717. Why use OpenFeign with Eureka?
      → Hint: auto-find service URLs

718. Why does Spring Cloud Gateway replace Zuul?
      → Hint: reactive + faster

719. Why use filters in Spring Cloud Gateway?
      → Hint: tweak requests on the way

720. Why centralize configuration?
      → Hint: settings stored in one magic drawer

721. Why use Spring Cloud Config Server?
      → Hint: share config across services

722. Why store config in Git?
      → Hint: version control superpowers

723. What is configuration refresh?
      → Hint: update settings without restart

724. Why avoid storing secrets in config files?
      → Hint: nosy people exist

725. Why use Vault or AWS Secrets Manager?
      → Hint: safe secret vaults

726. What is Hystrix for?
      → Hint: stop cascading failures

727. Why do we need circuit breakers?
      → Hint: avoid asking a dead service again and again

728. What is a fallback method?
      → Hint: plan B

729. Why is resilience key in microservices?
      → Hint: failures happen often

730. What is a retry mechanism?
      → Hint: try again politely

731. Why limit retries?
      → Hint: don’t annoy the server

732. Why implement rate limiting?
      → Hint: don’t let users go wild

733. Why use bulkheads?
      → Hint: isolate failures like ship sections

734. Why do microservices require distributed tracing?
      → Hint: follow a request’s journey

735. What does Sleuth do?
      → Hint: adds tracking IDs

736. What does Zipkin provide?
      → Hint: visual trace system

737. Why use correlation IDs?
      → Hint: track one request across many stops

738. Why is logging harder in microservices?
      → Hint: logs scattered everywhere

739. Why use centralized logging?
      → Hint: one big searchable bucket

740. Why use ELK stack or Loki?
      → Hint: read logs like a pro

741. Why is monitoring critical?
      → Hint: know when things fall apart

742. Why expose metrics?
      → Hint: health check for services

743. Why use Prometheus?
      → Hint: scrapes metrics automatically

744. Why pair Prometheus with Grafana?
      → Hint: pretty dashboards

745. Why are distributed systems hard?
      → Hint: everything fails everywhere all the time

746. What is network partition?
      → Hint: some services can’t reach each other

747. Why is CAP theorem important?
      → Hint: choose what matters: consistency or availability

748. What is eventual consistency?
      → Hint: everything becomes correct… eventually

749. Why avoid distributed transactions?
      → Hint: slow + messy

750. What is Saga pattern?
      → Hint: split big transaction into small steps

751. What is choreography in Saga?
      → Hint: services coordinate themselves

752. What is orchestration in Saga?
      → Hint: central conductor

753. Why use distributed locks carefully?
      → Hint: easy to mess things up

754. Why is idempotency crucial in microservices?
      → Hint: safe to retry

755. Why use message brokers?
      → Hint: async communication highway

756. Why choose Kafka?
      → Hint: fast, scalable, durable

757. Why choose RabbitMQ?
      → Hint: flexible routing

758. Why do messages need acknowledgment?
      → Hint: avoid losing them

759. Why must messages be durable?
      → Hint: survive crashes

760. Why avoid large messages in queues?
      → Hint: slow and heavy

761. Why use dead-letter queues?
      → Hint: catch bad messages

762. Why use event-driven architecture?
      → Hint: react to things happening

763. What is eventual consistency in event-driven systems?
      → Hint: updates flow through events

764. Why use event sourcing?
      → Hint: store events, not just results

765. Why is replaying events useful?
      → Hint: rebuild past states

766. Why must event payloads be versioned?
      → Hint: old apps still exist

767. Why do microservices need backward compatibility?
      → Hint: older clients still call you

768. Why do we need schema validation?
      → Hint: avoid broken messages

769. Why use schema registries?
      → Hint: track message formats

770. Why are synchronous microservices brittle?
      → Hint: chain reaction of failures

771. Why use timeout settings?
      → Hint: don’t wait forever

772. Why use bulkhead pattern?
      → Hint: isolate different workloads

773. Why use caching in microservices?
      → Hint: lighten the load

774. Why avoid excessive caching?
      → Hint: stale data trouble

775. Why use distributed cache?
      → Hint: share common results

776. Why use Redis for caching?
      → Hint: super-fast memory store

777. Why need cache eviction policies?
      → Hint: memory isn’t infinite

778. Why avoid sharing entity models between services?
      → Hint: hidden coupling

779. Why use API contracts?
      → Hint: define clear expectations

780. Why use OpenAPI/Swagger?
      → Hint: auto-generate documentation

781. Why need consumer-driven contracts?
      → Hint: prevent breaking clients

782. Why do microservices often adopt hexagonal architecture?
      → Hint: separate business from tech bits

783. Why is database migration tricky in microservices?
      → Hint: many services, many schemas

784. Why use Flyway/Liquibase?
      → Hint: version your database

785. Why prefer eventual consistency in distributed systems?
      → Hint: perfect is too slow

786. Why adopt CQRS?
      → Hint: split reading and writing

787. Why can CQRS be dangerous?
      → Hint: double the complexity

788. Why use distributed ID generators?
      → Hint: avoid collisions

789. Why use Snowflake ID pattern?
      → Hint: unique IDs at scale

790. Why is service mesh popular?
      → Hint: sidecars manage traffic

791. Why use Istio?
      → Hint: traffic control for microservices

792. Why use sidecar pattern?
      → Hint: put helpers outside main app

793. Why avoid too many microservices?
      → Hint: mini mayhem

794. Why use domain-driven design?
      → Hint: build services around business area

795. Why keep microservices small but not tiny?
      → Hint: balance matters

796. Why do microservices need strong observability?
      → Hint: more parts = more blind spots

797. Why choose gRPC over REST?
      → Hint: faster + typed contracts

798. Why avoid gRPC for public APIs?
      → Hint: browsers don’t play well

799. Why use API gateways for rate limiting and auth?
      → Hint: keep services clean

800. Why ensure microservices can scale independently?
      → Hint: one hungry service shouldn’t drag others

---

### ✔️ Batch 8 complete!

---

# ✅ **Batch 9 — Advanced Java, JavaFX, Caching, Reactive, Performance (Q801–Q900)**

### **One-Line Hard Questions + Tiny Hints**

801. Why is reflection powerful but dangerous?
      → Hint: poking private stuff isn’t always safe

802. Why does reflection slow applications?
      → Hint: sneaking around takes extra time

803. Why avoid reflection in performance-critical paths?
      → Hint: speed bumps everywhere

804. Why do frameworks like Spring depend on reflection?
      → Hint: dynamic magic needs backstage access

805. What are dynamic proxies used for?
      → Hint: create fake helpers at runtime

806. Why use InvocationHandler?
      → Hint: intercept method calls like a spy

807. Why can dynamic proxies only work with interfaces?
      → Hint: they need a contract to imitate

808. What problem does CGLIB solve?
      → Hint: proxy even without interfaces

809. Why must classes be non-final for CGLIB?
      → Hint: can’t extend a sealed box

810. Why is annotation processing used?
      → Hint: generate code before the show starts

811. Why choose compile-time annotation processing over runtime?
      → Hint: faster and safer

812. What is a marker annotation?
      → Hint: a tag with no words

813. Why use custom annotations?
      → Hint: add mini-rules for your system

814. Why can reflection break encapsulation?
      → Hint: enters locked rooms without permission

815. Why is JavaFX preferred over Swing in modern UI?
      → Hint: newer, fresher, shinier

816. Why does JavaFX support CSS styling?
      → Hint: UI fashion matters

817. Why use FXML for JavaFX layouts?
      → Hint: build UI without touching code

818. Why use Scene Builder?
      → Hint: drag-and-drop happiness

819. Why use property bindings in JavaFX?
      → Hint: auto-update values like magic

820. What is an observable list in JavaFX?
      → Hint: tells you when something changes

821. Why avoid heavy logic in JavaFX UI thread?
      → Hint: don’t freeze the screen

822. Why use Platform.runLater()?
      → Hint: tell UI thread to update later

823. Why use Worker and Task API?
      → Hint: do background stuff peacefully

824. Why is caching important in systems?
      → Hint: save expensive trips

825. Why is cache invalidation famously hard?
      → Hint: when to throw old stuff away?*

826. Why use LRU caching?
      → Hint: drop the least-visited guest first

827. Why avoid caching everything?
      → Hint: fridge space is limited

828. Why use Ehcache or Caffeine?
      → Hint: fast and friendly caching helpers

829. Why does Redis make a great distributed cache?
      → Hint: everything sits in memory waiting for you

830. Why choose Redis over Memcached?
      → Hint: more features, less drama

831. What is cache stampede?
      → Hint: everyone asks at once

832. How does cache pre-warming help?
      → Hint: prepare food before guests arrive

833. Why use write-through caching?
      → Hint: update cache + DB together

834. Why use write-behind caching?
      → Hint: update DB quietly later

835. Why does TTL matter for cache entries?
      → Hint: stale food spoils

836. What is reactive programming?
      → Hint: respond to things, don’t wait

837. Why does reactive avoid blocking?
      → Hint: no waiting in long queues

838. Why use Mono and Flux?
      → Hint: 1 item vs many items pipelines

839. Why does backpressure matter?
      → Hint: stop the firehose!*

840. Why is Reactor better for asynchronous flows?
      → Hint: chains events like dominoes

841. Why avoid mixing reactive and imperative code?
      → Hint: oil + water

842. What is a cold publisher?
      → Hint: starts producing when subscribed

843. What is a hot publisher?
      → Hint: keeps emitting no matter what

844. Why must reactive pipelines be immutable?
      → Hint: assemble once, use safely

845. What is flatMap used for in Reactor?
      → Hint: explode things into more things

846. What is concatMap used for?
      → Hint: one after another, neatly

847. What does publishOn do?
      → Hint: change the execution chair

848. What does subscribeOn do?
      → Hint: change where subscription begins

849. Why is debugging reactive code difficult?
      → Hint: invisible threads everywhere

850. Why enable Reactor debug mode?
      → Hint: see the hidden paths

851. Why is GC tuning important?
      → Hint: trash collector must be efficient

852. Why choose G1 GC?
      → Hint: fewer pauses for big apps

853. Why use ZGC?
      → Hint: ultra-low pause times

854. Why avoid excessive object creation?
      → Hint: GC gets overwhelmed

855. Why use JVM profiling tools?
      → Hint: spot the troublemakers

856. Why use JFR (Java Flight Recorder)?
      → Hint: a black box for your app

857. Why use VisualVM or JMC?
      → Hint: watch memory + CPU live

858. Why check thread dumps?
      → Hint: catch deadlocks red-handed

859. Why examine heap dumps?
      → Hint: find memory leaks

860. Why consider off-heap memory?
      → Hint: reduce pressure on GC

861. Why tune thread pools?
      → Hint: too few = slow, too many = chaos

862. Why avoid synchronized in high contention?
      → Hint: traffic jam

863. Why use locks sparingly?
      → Hint: people hate queues

864. Why use ReadWriteLock?
      → Hint: many readers, few writers

865. Why choose atomic classes?
      → Hint: thread-safe counters

866. Why monitor CPU utilization?
      → Hint: don’t overcook the machine

867. Why use connection pooling?
      → Hint: reusing connections saves time

868. Why test under realistic load?
      → Hint: surprises are bad

869. Why use JMH for microbenchmarks?
      → Hint: precise performance tests

870. Why avoid microbenchmarking naive code?
      → Hint: JVM optimizes behind your back

871. What is escape analysis?
      → Hint: decides if object stays in stack

872. What is inlining?
      → Hint: remove method call overhead

873. Why use -XX:+PrintCompilation?
      → Hint: see JIT magic

874. Why is false sharing harmful?
      → Hint: cache-line fights

875. How does @Contended help?
      → Hint: prevent sharing memory lanes

876. Why should buffers be reused?
      → Hint: allocations are expensive

877. Why is NUMA awareness important?
      → Hint: memory islands matter

878. Why use async database calls?
      → Hint: don’t block your threads

879. Why watch DB slow queries?
      → Hint: they become bottlenecks

880. Why index database fields carefully?
      → Hint: too many or too few both hurt

881. Why use query plans?
      → Hint: see how DB thinks

882. Why monitor thread contention?
      → Hint: threads fighting = slow app

883. Why choose proper data structures?
      → Hint: wrong tools slow you down

884. Why use StringBuilder for concatenation?
      → Hint: mutable = faster

885. Why avoid regex for simple parsing?
      → Hint: regex is heavy

886. Why pool expensive objects?
      → Hint: reuse saves energy

887. Why tune JVM heap size?
      → Hint: too big pauses, too small crashes

888. Why use async logging?
      → Hint: don’t block app threads

889. Why compress logs?
      → Hint: logs grow like wild plants

890. Why trace memory leaks early?
      → Hint: long-running apps suffer most

891. Why use defensive copies?
      → Hint: protect your data

892. Why avoid large synchronized blocks?
      → Hint: make them tiny

893. Why reuse thread-safe collections carefully?
      → Hint: safety isn’t always speed

894. Why profile before optimizing?
      → Hint: guesswork is wrong

895. Why monitor garbage collection pauses?
      → Hint: lag spikes!*

896. Why use batching for DB writes?
      → Hint: one big trip instead of many small

897. Why minimize serialization overhead?
      → Hint: converting data is expensive

898. Why prefer JSON over XML in most cases?
      → Hint: lighter and faster

899. Why consider binary serialization (like Kryo)?
      → Hint: super-fast data transport

900. Why test performance continuously?
      → Hint: today’s fast can become tomorrow’s slow

---

### ✔️ Batch 9 complete!

---

# ✅ **Batch 10 (Q901–Q1000): Advanced Microservices, Security, Testing, DevOps, Cloud & Java 17+**

### **One-Line Hard Questions + Tiny, Simple, Slightly Funny Hints**

901. Why is the Saga pattern essential for distributed transactions?
      → Hint: many small dances instead of one big dangerous dance

902. Why do Sagas avoid two-phase commit?
      → Hint: 2PC is slow and cranky

903. Why use CQRS in large systems?
      → Hint: reading and writing don’t get along

904. Why is event sourcing useful?
      → Hint: remember every little thing

905. Why is event replay powerful?
      → Hint: rewind your system’s life

906. Why must events be immutable?
      → Hint: no editing history books

907. Why use Outbox pattern?
      → Hint: send events safely from DB

908. Why use Change Data Capture?
      → Hint: watch DB changes live

909. Why does distributed tracing become critical at scale?
      → Hint: too many places to hide bugs

910. Why use correlation IDs in microservices?
      → Hint: track a request’s journey

911. Why implement API versioning?
      → Hint: keep old clients happy

912. Why choose backward compatibility?
      → Hint: don’t break old apps

913. Why adopt zero-downtime deployments?
      → Hint: users hate outages

914. Why use blue-green deployment?
      → Hint: swap versions like flipping a switch

915. Why use canary releases?
      → Hint: test on a tiny audience first

916. Why monitor release metrics?
      → Hint: catch disasters early

917. Why use distributed locks carefully?
      → Hint: they cause more trouble than they solve

918. Why avoid synchronous microservice chains?
      → Hint: domino effect of sadness

919. Why adopt asynchronous microservices?
      → Hint: no waiting in long queues

920. Why do microservices need strong observability?
      → Hint: more pieces = more mysteries

921. Why implement rate limiting in gateways?
      → Hint: stop greedy clients

922. Why use JWT for stateless auth?
      → Hint: no server memory needed

923. Why must JWT be short-lived?
      → Hint: lost tokens = trouble

924. Why store JWT secrets securely?
      → Hint: protect your magic key

925. Why validate JWT signature every time?
      → Hint: check ID cards properly

926. Why refresh tokens exist?
      → Hint: longer access without risk

927. Why does OAuth2 matter?
      → Hint: let people log in with someone else’s trust

928. Why use OAuth scopes?
      → Hint: limit what apps can do

929. Why use PKCE in OAuth2?
      → Hint: protect mobile apps

930. Why is SSO helpful?
      → Hint: one login to rule them all

931. Why must microservices handle CORS correctly?
      → Hint: browsers get picky

932. Why implement RBAC?
      → Hint: roles reduce chaos

933. Why use secrets vaults instead of environment variables?
      → Hint: env vars leak too easily

934. Why rotate secrets regularly?
      → Hint: don’t trust old keys

935. Why hash passwords instead of encrypting them?
      → Hint: one-way door

936. Why use bcrypt/argon2?
      → Hint: slow is safer

937. Why sanitize inputs?
      → Hint: keep bad stuff out

938. Why validate all external data?
      → Hint: trust no one

939. Why do Docker containers help consistency?
      → Hint: same box everywhere

940. Why use small Docker images?
      → Hint: faster to ship

941. Why use multi-stage Docker builds?
      → Hint: slim and clean output

942. Why avoid running containers as root?
      → Hint: too much power

943. Why use container health checks?
      → Hint: see if the app is alive

944. Why isolate containers with namespaces?
      → Hint: separate playgrounds

945. Why use Kubernetes for orchestration?
      → Hint: babysits containers at scale

946. Why use Deployments in Kubernetes?
      → Hint: manage app versions

947. Why use StatefulSets?
      → Hint: keep order and identity

948. Why use ConfigMaps?
      → Hint: store non-secret settings

949. Why use Secrets in Kubernetes?
      → Hint: hide passwords from snoopers

950. Why use Horizontal Pod Autoscaling?
      → Hint: add more workers when busy

951. Why use liveness and readiness probes?
      → Hint: check health and preparedness

952. Why adopt service mesh?
      → Hint: automatic traffic magic

953. Why use Istio sidecars?
      → Hint: outsource networking chores

954. Why monitor pod resource usage?
      → Hint: avoid noisy neighbors

955. Why limit CPU/memory per pod?
      → Hint: stop greedy apps

956. Why use Helm charts?
      → Hint: reusable deployment recipes

957. Why use CI/CD pipelines?
      → Hint: automate everything

958. Why run unit tests in CI?
      → Hint: save yourself embarrassment

959. Why run integration tests?
      → Hint: check teamwork

960. Why run contract tests in microservices?
      → Hint: don’t break your clients

961. Why run performance tests regularly?
      → Hint: slow creeps in quietly

962. Why use feature flags?
      → Hint: turn features on/off easily

963. Why prefer immutable infrastructure?
      → Hint: no manual fiddling

964. Why use Infrastructure as Code?
      → Hint: treat servers like code

965. Why adopt Terraform?
      → Hint: declare cloud resources cleanly

966. Why use AWS IAM carefully?
      → Hint: permissions can cause disasters

967. Why prefer least-privilege access?
      → Hint: give tiny keys, not master keys

968. Why use VPCs in cloud environments?
      → Hint: private playground

969. Why deploy across multiple AZs?
      → Hint: avoid single-zone sadness

970. Why replicate data across regions?
      → Hint: global resilience

971. Why use load balancers in cloud apps?
      → Hint: distribute the party

972. Why use autoscaling groups?
      → Hint: add/remove servers automatically

973. Why use CloudWatch/Stackdriver/Monitor?
      → Hint: eyes on everything

974. Why rely on SQS/Kinesis/PubSub?
      → Hint: handle high-volume messages

975. Why use CDN for static content?
      → Hint: deliver from nearest place

976. Why prefer object storage like S3?
      → Hint: infinite and cheap

977. Why encrypt data at rest?
      → Hint: protect sleeping data

978. Why encrypt data in transit?
      → Hint: protect traveling data

979. Why adopt multi-tenancy carefully?
      → Hint: separate customers safely

980. Why use sharding?
      → Hint: split data like pizza slices

981. Why use read replicas?
      → Hint: handle heavy reading

982. Why choose NoSQL for large-scale reads?
      → Hint: flexible and fast

983. Why choose SQL for strong consistency?
      → Hint: strict rules, clean data

984. Why use distributed caching like Redis cluster?
      → Hint: don’t overload databases

985. Why use message-driven APIs?
      → Hint: asynchronous wins

986. Why track API performance metrics?
      → Hint: find slow endpoints

987. Why version Docker images explicitly?
      → Hint: “latest” is a liar

988. Why regularly prune unused containers/images?
      → Hint: don’t fill the house with junk

989. Why keep JVM inside containers tuned?
      → Hint: containers lie about memory

990. Why use JDK Flight Recorder in production?
      → Hint: low-overhead detective

991. Why upgrade to Java 17+?
      → Hint: faster, cleaner, safer

992. Why use records in Java 17?
      → Hint: tiny classes without boilerplate

993. Why use sealed classes?
      → Hint: limit who can inherit

994. Why use pattern matching in Java 17?
      → Hint: cleaner type checks

995. Why use text blocks?
      → Hint: write multiline strings peacefully

996. Why prefer switch expressions?
      → Hint: shorter and safer

997. Why embrace ZGC/Shenandoah?
      → Hint: ultra-low pauses

998. Why use virtual threads (Project Loom)?
      → Hint: millions of friendly threads

999. Why adopt structured concurrency (Loom)?
      → Hint: manage threads like a tidy parent

1000. Why does Java continue to evolve rapidly?
      → Hint: staying cool in a fast world

---

### 🎉 All 1000 questions completed!
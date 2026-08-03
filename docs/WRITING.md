# Documentation Standard

## Write for a reader with a task

Open with the problem the reader is trying to solve. Do not open with a module inventory or a definition that gives no
reason to care.

Connect new material to knowledge established earlier. After the first-plugin tutorial, `main`, `State`, and server tools
can be introduced through what that plugin already did.

Order sections by the reader's likely path. Put the primary runtime path before secondary lifecycle or administrative APIs.

## Build a narrative

Technical narrative carries the reader from one state of understanding to the next:

1. Establish what the reader has already built or learned.
2. Introduce the next problem in terms of that existing knowledge.
3. Present the new concept because it solves that problem.
4. Show the concept in use.
5. Explain what changed and what the reader can do next.

Do not confuse narrative with decoration. A transition earns its place when it preserves cause and effect, recalls a relevant
earlier result, or sets up the next decision. A sentence that merely announces a section still has no value.

Do not confuse concision with dryness. Definitions and tables provide facts, but prose must connect those facts into the
reader's workflow. A page should not read like an API inventory with examples inserted between entries.

Let cause and effect give the page movement. Explain why the reader encounters a concept, then explain what the concept
changes. Do not invent tension to make the explanation feel like a story.

Do not make the prose cinematic. Narrative should sound like a thoughtful engineer explaining why the system works this way,
not like a dramatic scene. Avoid inflated framing such as silence, suspense, or stalled motion when plain technical language
will do.

Do not try to make headers clever. A good header names the subject cleanly. Prefer `Raw reasoning text` or `Built-in
summary plugin` over cutesy or theatrical titles.

Avoid flat sequences of declarations such as "X is this. Then Y does that." Show why Y becomes necessary after X, and make
that relationship explicit in the transition.

Keep the page airy. Use short paragraphs, and give examples and important distinctions their own space. Dense prose can make
a clear explanation difficult to follow even when every sentence is accurate.

Whitespace alone is not enough. Vary sentence length and connect the paragraphs so the page still reads as one explanation,
not as a stack of isolated facts.

Do not isolate an ordinary definition or transition in a one-line paragraph to manufacture emphasis. Whitespace should give
real ideas room, not make a routine sentence sound dramatic.

Do not solve flat writing by making the page too short. If the reader needs more context, add the missing context. Brevity is
not a substitute for a complete explanation.

Tutorials should end by locating the reader in the larger system. A `Continue learning` section can recap what the reader
can now build and offer a small number of next paths based on plausible goals. Each path needs enough context to feel like a
continuation, not a bare link followed by a destination summary.

Do not compress a learning path into one flat sentence to make it shorter. Give the path room to connect the tutorial to the
next idea. For example:

> If your next plugin adds another server-owned capability, continue with Server tools. It starts from the `ServerTool` used
> here and develops it beyond the no-argument example.

This is better than:

> If the next capability is another model-callable Python function, continue with Server tools to add arguments and
> validation.

The first version preserves the reader's momentum and relates the next guide to work they just completed. The second reduces
that progression to a link plus a feature list.

## Write with life, not ornament

Ornamental writing uses words without adding meaning. Removing ornamental writing does not mean removing voice, energy, or
high-level explanation.

Prefer the outcome the reader cares about when orienting them. "It gives the model a new capability" is clear and useful.
Do not replace it with a mechanical sentence such as "the model can call Python code owned by the plugin." Mechanics belong
where the reader needs to understand implementation or constraints, not where the writing should establish purpose.

Remove only the part that carries no information. In "it gives the model a new capability while plap continues to run the
response loop," the first clause is useful and the second clause is redundant. Keep the first clause instead of rewriting
the whole sentence into dry exposition.

Use active verbs and reader-visible consequences. Let the prose communicate why a feature is powerful, not only how its
objects are wired together.

Avoid vague idioms such as "ready to stand behind." Name the actual condition: an answer may still fail validation, call a
tool, receive review feedback, or be replaced.

Read narrative prose aloud. Awkward cadence, repeated sentence shapes, and technically correct but lifeless phrasing are
editing problems, not matters of taste to ignore.

Technical accuracy and lively writing support each other. State the motivating idea plainly, then add mechanics where they
answer a real question raised by that idea.

Rhetorical questions can give a page movement when they name a question the reader is already asking, such as "What
happened?" or "Which request crosses the boundary?" Answer the question immediately. Do not turn every heading into a
question or use rhetorical questions as decoration.

Do not turn an author's exploratory question into documentation automatically. A question such as "is this push or pull?"
may be asking for implementation research, not proposing a reader-facing section. Use the answer to improve accuracy, then
write from the reader's problem unless the distinction changes how the reader uses the feature.

Lead with experience and purpose. Add mechanism only when it explains a behavior, constraint, or choice the reader cares
about. Do not let an internal pipeline take over a page whose real subject is the user experience it enables.

Do not slip into product-copy language. If a built-in plugin is an example implementation, say what it does. Do not say it
"helps" unless you are directly describing how it helps the reader solve their task.

Avoid stock contrasts such as "It is not X. It is Y." State the idea naturally. Use an explicit contrast only when the
reader must understand a real distinction between two plausible interpretations.

## Explain concepts before APIs

Explain what a concept changes before listing its fields or methods. Introduce API names at the point where they solve the
stated problem.

Keep distinct concepts separate. Examples include:

- A thread stores an isolated message history; `threads.active` controls client-tool participation and main publication.
- `state.memory` carries response-level plugin data; `ChatMessage.memory` belongs to one message.
- Routing retries provider failures; completion retries reject unusable model results.
- Server tools add model-callable functions; response hooks modify existing execution.

## Include the reason and the consequence

Do not state a mechanism without its effect. If code removes `main` from `threads.active`, explain that the main loop stops
and unpublished main output remains private.

Do not claim ownership that the API does not enforce. If plugins can call `save_progress()`, do not say core decides when
progress is saved.

Mark request and lifecycle boundaries explicitly. If state is persisted at the end of one request and consumed in another,
do not place the two mutations next to each other as if they form an immediate pair. State what ends the first request, what
is restored later, and what condition allows the second mutation.

When "request" could mean traffic on both sides of the system, define qualified terms before describing the flow. Use names
such as "model request," "Responses request," and "Responses continuation" instead of "next request." State who sends each
request and who receives it.

Every technical claim must match the current code. Read the implementation and tests before describing behavior.

## Remove empty prose

Delete sentences that only announce the page, praise an abstraction, or tell the reader to use an unspecified subset.

Avoid phrases such as:

- "This page documents..."
- A bare "Continue with..." link that does not continue the reader's current work.
- "Choose the narrowest..."
- "Use only the part you need."
- "The objects have one job each."
- "Behavior belongs at this boundary."

A link should be attached to a concrete fact. For example: "A response hook can change how plap builds the next model
request" is useful; "continue with hooks" is not.

Navigation must name a concrete reason to follow the link. Avoid vague promises such as "the complete contract" unless the
target is specifically a formal input/output contract.

Concrete does not mean exhaustive. Navigation should connect the reader's current result to the next problem. Do not dump a
comma-separated list of features from the destination page.

## Do not dump information

Do not substitute an inventory for an explanation. Tables and lists are useful references after the relationship between
their entries is clear; they are not connective prose.

Keep each transition at one level of abstraction. If the reader is choosing an extension model, explain the difference
between the models. Do not jump down into unrelated method names, lifecycle details, and edge cases to justify a link.

Synthesize related facts before listing details. More facts make the writing better only when they help answer the current
question.

Implementation research is a filter for accuracy, not a source of material to copy into the page. After tracing the code,
keep only the facts needed to explain the reader-visible behavior or a decision the reader must make.

## State each fact once

Give each concept one primary explanation. Other pages may provide enough context to use a link, but must not repeat the
same inventory, lifecycle, or selection guide.

Examples should add information. Do not show an incomplete class and then repeat its argument and return code in separate
fragments. Prefer one complete example followed by explanation of the non-obvious lines.

## Give examples a reason to exist

Introduce an example with the problem it solves. The reader should know what the plugin is trying to accomplish before seeing
the hook, field, or method chosen for it.

Explain why the selected API fits the task. Compare it with another API only when the reader might plausibly choose between
them. Do not invent a contrast to create a transition.

Give every example its own concrete scenario. When examples teach different patterns, make that difference clear through the
scenarios instead of announcing that "the examples below show two patterns."

On an informational page with several examples, group them under an explicit `Examples` heading. The heading separates
conceptual reference material from applied code without adding transition prose.

After the code, explain the control flow that is not obvious from reading it. Do not restate each line.

A reference table can summarize an API, but it cannot carry the page by itself. Connect reference material to concrete
decisions, causes, and effects.

## Use direct language

Name the concrete request, message, call, result, or state change. Avoid vague substitutes such as "line of work,"
"materialize," "owns the boundary," or "the machinery."

Keep sentences grammatically simple. More detail is welcome when each sentence adds a fact, reason, constraint, or
consequence.

Edit surgically. When one clause is empty, remove that clause without flattening the useful sentence around it.

Do not optimize for the fewest sentences. Two connected sentences are better than one compressed sentence when the second
sentence carries the reader from existing knowledge into the next concept.

Introduce named components before referring to them. Write "the built-in `advisor` review plugin" before using "advisor"
as an example.

## Use diagrams for structure

Use a diagram when it clarifies nesting, branching, concurrency, or separate histories. Do not turn a linear sentence or a
field list into a staircase of arrows.

Use tree diagrams for hierarchy. Use tables for API mappings. Use numbered lists for sequences. Keep labels short and avoid
decorative connectors.

## Keep examples honest

Examples must be complete enough to support the claim around them. Mark fragments as fragments and do not present
placeholder behavior as a working implementation.

Use current imports and signatures. Compile Python examples and validate shell commands, links, and heading anchors before
finishing a documentation change.

## Review checklist

Before accepting a page, verify:

1. The opening explains why the reader needs the subject.
2. The section order follows the reader's task rather than the source tree.
3. Every API is introduced after its concept.
4. Every mechanism includes its relevant consequence.
5. No fact is duplicated in another section or page.
6. No proper noun appears without an introduction.
7. No sentence exists only to announce, transition, or decorate.
8. Diagrams encode structure that prose or a table would express less clearly.
9. Examples are accurate, complete, and verified.
10. The page carries the reader from prior knowledge through a problem, solution, and consequence.
11. The prose states reader-visible outcomes before dropping into mechanics.
12. Removing ornament has not made the writing dry or lifeless.
13. Every example begins with a concrete reason for choosing that API.
14. Multiple examples each have a concrete scenario and demonstrate distinct behavior without a generic announcement.
15. Navigation continues the reader's current work instead of listing destination features.
16. Tables and lists support an explanation rather than replace it.
17. Examples that cross requests or lifecycle phases label those boundaries explicitly.
18. Every technical claim matches the current implementation and tests.

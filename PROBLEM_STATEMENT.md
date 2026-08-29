# AI-Powered Universal Chunking and Information Extraction

## Background

Organizations generate and store enormous amounts of information across multiple formats, including text documents, PDFs, images, audio recordings, videos, presentations, spreadsheets, and other digital content.

This information is increasingly being used to build AI systems such as RAG applications, AI Agents, enterprise search platforms, knowledge management systems, and intelligent automation solutions.

Before these systems can effectively use the information, the raw content must be understood, processed, and transformed into meaningful knowledge.

However, different types of content contain information in fundamentally different ways. A PDF may contain document hierarchy, paragraphs, tables, images, and references. An image may contain text, objects, relationships, and visual context. A video may contain speech, characters, actions, scenes, events, and temporal information.

## The Problem

Current information processing and chunking approaches often treat different types of content as plain text or apply the same fixed chunking strategy to every input.

This can result in:
- Loss of contextual meaning
- Broken relationships between information
- Loss of document structure
- Separation of related information
- Poor retrieval quality
- Incomplete knowledge extraction
- Ineffective downstream AI performance

For example, simply extracting text from a PDF may lose the relationship between a heading, its section, a table, and an associated image.

Similarly, transcribing a video alone may fail to capture what is happening visually, who is involved, what actions are being performed, and how events relate to one another over time.

The challenge is therefore not simply to split content into smaller chunks. The challenge is to intelligently understand different forms of information, extract meaningful knowledge, preserve relevant context and relationships, and transform the content into reusable knowledge units.

## Target Users and Organizations

This problem is relevant to organizations and teams working with large volumes of diverse information, including:
- Enterprise knowledge management teams
- AI and RAG product teams
- Research organizations
- Educational institutions
- Media and content organizations
- Legal and compliance teams
- Healthcare and pharmaceutical organizations
- Data and AI engineering teams
- Organizations building AI Agents

The solution should have the potential to become a reusable product or platform capable of processing information across different organizational domains.

## Challenge

Build an AI-powered system capable of intelligently extracting and chunking information from diverse input types.

The system should analyze an input, understand the type and structure of information it contains, extract relevant information, identify contextual relationships, and generate meaningful chunks suitable for downstream AI systems.

The solution should adapt its extraction and chunking approach based on the nature and complexity of the input.

Participants are not required to support every possible input type. Teams may choose the formats and difficulty levels they want to address and are encouraged to innovate beyond the examples provided.

The objective is to build a system that transforms raw content into contextually meaningful, structured, and reusable knowledge.

## Expected Capabilities

### Understand the Input
Identify the type, structure, and relevant components of the provided information.

### Extract Meaningful Information
Extract information appropriate to the type of input rather than applying the same extraction method to all formats.

### Preserve Context
Ensure that generated chunks retain sufficient information to remain meaningful when processed independently.

### Understand Structure
Identify relevant relationships and structures within the content, such as Heading → Section → Paragraph, Table → Row → Column, Speaker → Statement, Character → Action, Scene → Event, Image → Caption, and Event → Timestamp.

### Perform Intelligent Chunking
Generate chunks based on semantic meaning, structure, context, relationships, or temporal information rather than relying exclusively on fixed token or character limits.

### Generate Useful Metadata
Produce metadata that improves retrieval and downstream usage where appropriate.

### Support Machine-Usable Outputs
Generate outputs that can potentially be consumed by RAG systems, AI Agents, enterprise search systems, knowledge bases, and analytics platforms.

## Input Types and Difficulty Levels

The following levels are intended as guidance for increasing complexity. Participants may choose one or more levels based on their approach.

### Level 1: Text and Structured Content

**Possible inputs may include:**
- Plain text
- Markdown
- Articles
- Emails
- Web content
- Structured documents

**Potential information to understand and extract may include:**
- Topics
- Sections
- Context
- Entities
- Relationships
- Metadata
- Information boundaries

### Level 2: Complex Documents

**Possible inputs may include:**
- PDFs
- Word documents
- Research papers
- Reports
- Presentations
- Technical documentation

**Potential information to understand and extract may include:**
- Titles
- Headings
- Sections and subsections
- Paragraphs
- Tables
- Lists
- Images
- Captions
- References
- Document hierarchy
- Relationships between content elements

The challenge is to preserve the structure rather than treating the entire document as a single stream of text.

### Level 3: Images and Visual Content

**Possible inputs may include:**
- Images
- Scanned documents
- Photographs
- Infographics
- Diagrams
- Screenshots

**Potential information to understand and extract may include:**
- Text
- Objects
- Visual entities
- Relationships
- Context
- Scene information
- Layout
- Spatial relationships
- Metadata

The solution should demonstrate how visual information and extracted textual information can be connected where relevant.

### Level 4: Audio Content

**Possible inputs may include:**
- Conversations
- Meetings
- Interviews
- Lectures
- Voice recordings
- Podcasts

**Potential information to understand and extract may include:**
- Transcription
- Speakers
- Speaker transitions
- Topics
- Context
- Questions and answers
- Decisions
- Actions
- Events
- Temporal relationships

The solution may explore how conversational context can be preserved when generating chunks.

### Level 5: Video Content

**Possible inputs may include:**
- Meetings
- Educational videos
- Tutorials
- Interviews
- Product demonstrations
- Operational recordings

**Potential information to understand and extract may include:**
- Speech transcription
- Characters or people
- Visual entities
- Objects
- Actions
- Events
- Scene changes
- Context
- Temporal relationships
- Important moments

A meaningful chunk may combine information such as: What was said + what happened visually + relevant entities + actions + temporal context.

Participants are free to define their own approach for representing and chunking multimodal information.

### Advanced Level: Multimodal Content

Advanced solutions may support inputs containing multiple forms of information simultaneously.
- PDFs containing text, tables, and images
- Videos containing speech and visual events
- Presentations containing text, diagrams, and images
- Meetings containing audio, video, and chat
- Web pages containing text, images, tables, and media

The challenge at this level is to preserve relationships across different types of information.

## Expected Outputs

The solution should produce meaningful chunks and extracted information that can be used by downstream systems.

A generated knowledge unit may include:
- Extracted content
- Context
- Metadata
- Structural information
- Source information
- Relationships
- Temporal information
- References to associated visual or audio content

**Content:** The extracted meaningful information.

**Context:** Information required to understand the content independently.

**Metadata:** Information such as source, hierarchy, timestamps, entities, relationships, and input type.

The output format is intentionally open. Participants may design their own representation and data model.

## Scope

Participants are free to decide:
- Which input types to support
- Which difficulty level to address
- What information should be extracted
- How chunks are represented
- Which AI models or tools are used
- The chunking strategy
- The metadata structure
- The system architecture
- Whether multiple AI models or agents are used
- Teams may use real, synthetic, or publicly available data where appropriate.

## Constraints

1. **Meaningful AI Usage:** The solution should demonstrate why intelligent processing is required for the selected problem. Solutions should go beyond simply applying fixed-size text splitting or basic file conversion.

2. **Extraction and Chunking Should Be Connected:** The extracted structure, context, or information should meaningfully influence how chunks are generated.

3. **Context Preservation:** The solution should attempt to minimize unnecessary loss of context and important relationships.

4. **Appropriate Processing for Different Inputs:** Teams supporting multiple input types should demonstrate how their approach adapts to differences in structure and information.

5. **Machine-Usable Output:** Generated knowledge should be structured so that it can be meaningfully consumed by a downstream AI or information system.

6. **Efficiency:** Participants should consider LLM consumption, token usage, model selection, processing time, cost, and scalability. Using an LLM or large model for every processing step without considering efficiency should not automatically be considered a better solution.

7. **Innovation Freedom:** Participants are not restricted to a specific model, framework, chunking algorithm, architecture, or technology.

## Success Criteria and Evaluation

| Criteria | What It Measures |
|----------|-----------------|
| Chunk Quality | How meaningful, coherent, and useful the generated chunks are |
| Extraction Quality | Accuracy and usefulness of extracted information |
| Context Preservation | Ability to preserve meaning and relationships |
| Adaptability | Ability to handle different structures or input characteristics |
| AI Necessity | Whether AI provides meaningful value beyond basic automation |
| Technical Quality | Architecture, implementation, and robustness |
| Efficiency | Effective use of models, computation, cost, and latency |
| Innovation | Novelty of the extraction or chunking approach |
| Scalability | Potential to process large volumes of information |
| Product Potential | Ability to evolve into a practical organizational product |
| Demo Quality | Ability to clearly demonstrate the working solution |

## Freedom to Innovate

Participants are free to choose their own technical approach.

Solutions may use any relevant combination of:
- LLMs
- Multimodal models
- OCR
- Speech-to-text
- Computer Vision
- Embedding models
- Semantic chunking
- Hierarchical chunking
- Contextual chunking
- Temporal chunking
- Knowledge graphs
- Machine learning
- Custom algorithms
- Agentic AI

The listed technologies and input types are examples only and should not limit innovation.

Participants are encouraged to introduce new input types, extraction methods, chunking approaches, metadata strategies, or downstream applications that improve the value of the solution.

## Core Objective

Given any form of information, intelligently understand what it contains, extract the meaningful knowledge, preserve its context and relationships, and transform it into high-quality chunks that can be effectively used by AI systems.

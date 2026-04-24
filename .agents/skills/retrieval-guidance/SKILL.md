---
name: retrieval-guidance
description: Use this skill when answering questions about the indexed PDF collection so you retrieve supporting pages, optionally filter by domain, and cite document names and page numbers.
allowed-tools: retrieve_pages
---

# retrieval-guidance

## Overview

This skill guides the agent to use the retrieval tool before making claims about the indexed documents.

## Instructions

1. If the user asks a question about the indexed corpus, call `retrieve_pages` before answering.
2. If the user or context indicates a relevant domain, pass that domain filter to the tool.
3. Base the answer on the strongest retrieved pages.
4. Cite the document name and page number when you reference retrieved evidence.
5. If retrieval returns no useful evidence, say that clearly and avoid guessing.

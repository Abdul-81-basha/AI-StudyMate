\# AI StudyMate



\### Agentic RAG-Based Laboratory Manual Analyzer



AI StudyMate is a Generative AI application designed to help engineering students understand laboratory manuals, prepare for experiments, generate viva questions, and create personalized study plans.



The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded laboratory manuals before generating answers using Google Gemini.



\---



\## 🚀 Live Application



\### Frontend

https://ai-study-mate-n635wv8zg-kare8.vercel.app



\### Backend API

PASTE\_YOUR\_RENDER\_URL\_HERE



\### Backend Health

PASTE\_YOUR\_RENDER\_URL\_HERE/health



\---



\## 🎯 Problem Statement



Engineering students often have to study lengthy laboratory manuals containing experiment objectives, theory, procedures, algorithms, source code, outputs, and viva questions.



Finding relevant information quickly during laboratory preparation can be difficult.



AI StudyMate solves this problem by allowing students to upload their laboratory manuals and interact with them using natural language.



\---



\## 👥 Target Users



\- Engineering students

\- CSE students

\- Laboratory exam candidates

\- Laboratory instructors

\- Students preparing for viva examinations



\---



\## ✨ Key Features



\- 📄 Upload laboratory manuals in PDF format

\- 🔍 Retrieval-Augmented Generation (RAG)

\- 🧠 Google Gemini-powered answers

\- 📚 Experiment-specific information retrieval

\- 📑 Source and page references

\- 💬 Interactive AI chat

\- 🎓 Viva question generation

\- 📅 Study plan generation

\- ⚡ Fast semantic retrieval

\- 💻 Responsive and modern user interface

\- 🔐 Backend-only API key handling



\---



\## 🏗️ System Architecture



```text

&#x20;                   AI STUDYMATE

&#x20;                        |

&#x20;                        v

&#x20;             ┌─────────────────────┐

&#x20;             │   React + Vite UI   │

&#x20;             │       Vercel        │

&#x20;             └──────────┬──────────┘

&#x20;                        |

&#x20;                     REST API

&#x20;                        |

&#x20;                        v

&#x20;             ┌─────────────────────┐

&#x20;             │   FastAPI Backend   │

&#x20;             │       Render        │

&#x20;             └──────────┬──────────┘

&#x20;                        |

&#x20;             ┌──────────┴──────────┐

&#x20;             |                     |

&#x20;             v                     v

&#x20;     ┌────────────────┐    ┌────────────────┐

&#x20;     │ Local Vector   │    │ Google Gemini  │

&#x20;     │ Store / SQLite │    │ LLM + Embedding│

&#x20;     └────────┬───────┘    └───────┬────────┘

&#x20;              ^                    |

&#x20;              |                    v

&#x20;              |              AI Generated

&#x20;              |                Response

&#x20;              |                    |

&#x20;              └──── RAG Context ──┘


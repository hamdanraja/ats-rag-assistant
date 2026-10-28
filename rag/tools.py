import re
from difflib import get_close_matches
from langchain_core.tools import tool

def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()

def candidate_name(candidate):
    if candidate.get("candidate_name"):
        return str(candidate["candidate_name"]).strip()
    name = str(candidate.get("filename", ""))
    name = re.sub(r"\.(pdf|docx|doc|txt)$", "", name, flags=re.I)
    name = re.sub(r"[_-]+", " ", name)
    name = re.sub(r"\b(resume|cv)\b", "", name, flags=re.I)
    return re.sub(r"\s+", " ", name).strip()

def candidate_resume_text(candidate):
    return str(candidate.get("resume_text") or "")

IGNORED = {
    "resume", "candidate", "accountant", "engineer",
    "developer", "data", "scientist", "profile",
}

def candidate_tokens(candidate):
    return [
        token for token in normalize(candidate_name(candidate)).split()
        if len(token) >= 4 and token not in IGNORED
    ]

def resolve_candidate(name, candidates):
    needle = normalize(name)
    if not needle:
        return None
    exact = [
        candidate for candidate in candidates
        if needle in {
            normalize(candidate_name(candidate)),
            normalize(candidate.get("filename", "")),
        }
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None
    matches = []
    for candidate in candidates:
        values = [
            normalize(candidate_name(candidate)),
            normalize(candidate.get("filename", "")),
        ]
        if any(f" {needle} " in f" {value} " for value in values):
            matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return None
    matches = [
        candidate for candidate in candidates
        if needle in candidate_tokens(candidate)
    ]
    return matches[0] if len(matches) == 1 else None

def candidate_report(candidate):
    return {
        "candidate": candidate_name(candidate),
        "filename": candidate.get("filename"),
        "rank": candidate.get("rank"),
        "score": candidate.get("score"),
        "skill_score": candidate.get("skill_score"),
        "experience_score": candidate.get("experience_score"),
        "qualification_score": candidate.get("qualification_score"),
        "description_score": candidate.get("description_score"),
        "matched_skills": candidate.get("matched_skills", []),
        "missing_skills": candidate.get("missing_skills", []),
        "total_experience": candidate.get("experience_years"),
        "relevant_experience": candidate.get("relevant_experience_years"),
        "experience_status": candidate.get("experience_status"),
        "qualifications": candidate.get("matched_qualifications", []),
        "education": candidate.get("education_evidence", []),
        "career_history": candidate.get("career_history", []),
    }

STOPWORDS = {
    "what", "which", "where", "when", "who", "does", "did",
    "the", "and", "for", "with", "about", "tell", "me",
    "his", "her", "their", "candidate", "resume", "show",
    "give", "have", "has", "from", "that", "this",
    "are", "was", "were",
}

def lexical_search(query, documents, k=5):
    terms = {
        word for word in normalize(query).split()
        if len(word) > 2 and word not in STOPWORDS
    }
    results = []
    for doc in documents:
        text = normalize(doc.page_content)
        score = sum(
            1 for term in terms
            if re.search(rf"\b{re.escape(term)}\b", text)
        )
        if score:
            results.append((score, doc))
    results.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in results[:k]]

class RagTools(list):
    def __init__(self, tools, candidates, vectorstore, documents, job_title, job_description):
        super().__init__(tools)
        self.candidates = candidates
        self.vectorstore = vectorstore
        self.documents = documents
        self.job_title = job_title
        self.job_description = job_description
        self.tool_map = {item.name: item for item in tools}

def create_rag_tools(vectorstore, candidates, job_title, job_description, documents=None):
    if documents is None:
        documents = getattr(vectorstore, "_rag_documents", [])

    @tool
    def get_ats_data(candidate: str = "") -> dict:
        """Get exact ATS rankings, scores, matched skills, missing skills, qualifications and experience,career history, employers, current positions and previous employment information. Use an empty candidate string for all candidates."""
        if candidate.strip():
            target = resolve_candidate(candidate, candidates)
            if target is None:
                return {
                    "error": "Candidate not found or ambiguous.",
                    "available_candidates": [candidate_name(c) for c in candidates],
                }
            return {"candidates": [candidate_report(target)]}
        ranked = sorted(
            candidates,
            key=lambda c: (
                c.get("rank") is None,
                c.get("rank") if isinstance(c.get("rank"), (int, float)) else float("inf"),
            ),
        )
        return {
            "job_title": job_title,
            "candidates": [candidate_report(c) for c in ranked],
        }

    @tool
    def search_resumes(query: str, candidate: str = "") -> dict:
        """Search original resumes for projects, work experience, technical skills, education, certifications, achievements and employment history. Optionally restrict to a candidate."""
        target = None
        if candidate.strip():
            target = resolve_candidate(candidate, candidates)
            if target is None:
                return {"error": "Candidate not found or ambiguous."}

        def allowed(doc):
            if doc.metadata.get("source_type") != "resume":
                return False
            if target is None:
                return True
            return doc.metadata.get("filename") == target.get("filename")

        selected_documents = [doc for doc in documents if allowed(doc)]
        if not selected_documents:
            return {"evidence": []}
        semantic = vectorstore.similarity_search(query, k=5, filter=allowed)
        lexical = lexical_search(query, selected_documents, k=5)
        unique = {}
        for doc in lexical + semantic:
            key = (doc.metadata.get("filename"), doc.page_content)
            if key not in unique:
                unique[key] = doc
        return {
            "evidence": [
                {
                    "filename": doc.metadata.get("filename"),
                    "candidate": doc.metadata.get("candidate"),
                    "text": doc.page_content,
                }
                for doc in list(unique.values())[:8]
            ]
        }

    @tool
    def search_job_description(query: str) -> dict:
        """Search the active job description for responsibilities, requirements, qualifications, skills and experience requirements."""
        def allowed(doc):
            return doc.metadata.get("source_type") == "job_description"

        selected_documents = [doc for doc in documents if allowed(doc)]
        if not selected_documents:
            return {"job_title": job_title, "evidence": []}
        semantic = vectorstore.similarity_search(query, k=4, filter=allowed)
        lexical = lexical_search(query, selected_documents, k=4)
        unique = {}
        for doc in lexical + semantic:
            if doc.page_content not in unique:
                unique[doc.page_content] = doc
        return {
            "job_title": job_title,
            "evidence": [
                {
                    "filename": "JOB_DESCRIPTION",
                    "text": doc.page_content,
                }
                for doc in list(unique.values())[:6]
            ],
        }

    return RagTools(
        tools=[get_ats_data, search_resumes, search_job_description],
        candidates=candidates,
        vectorstore=vectorstore,
        documents=documents,
        job_title=job_title,
        job_description=job_description,
    )
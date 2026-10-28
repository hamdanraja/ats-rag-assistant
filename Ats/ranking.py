import re
from datetime import date
from typing import Optional

def normalize(text: str) -> str:
    text = str(text or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def contains_term(text: str, term: str) -> bool:
    text, term = normalize(text), normalize(term)
    if not term:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text))

SKILL_ALIASES = {
    "microsoft excel": ["ms excel", "excel spreadsheets"],
    "quickbooks": ["quick books"],
    "accounts payable": ["account payable", "ap processing"],
    "accounts receivable": ["account receivable", "ar processing"],
    "bank reconciliation": ["bank reconciliations"],
    "general ledger management": ["general ledger", "ledger management", "gl management"],
    "financial reporting": ["financial statements", "financial reports"],
    "tax preparation": ["tax filing", "tax returns"],
    "bookkeeping": ["book keeping"],
    "artificial intelligence": ["ai"],
    "machine learning": ["ml"],
    "natural language processing": ["nlp"],
    "large language models": ["llm", "llms"],
    "retrieval augmented generation": ["rag"],
    "hugging face transformers": ["huggingface transformers", "transformers library"],
    "rest apis": ["rest api", "restful api"],
    "vector databases": ["vector database"],
}

NEGATION_PATTERN = re.compile(
    r"\b(no|not|never|without|lack|lacking|limited exposure to|unfamiliar with|no experience with|not experienced in)\b",
    re.IGNORECASE,
)

def clean_skills(raw_skills: list[str]) -> list[str]:
    result, seen = [], set()
    for skill in raw_skills:
        skill = re.sub(r"^(and|or)\s+", "", str(skill).strip(), flags=re.IGNORECASE)
        skill = skill.strip(" .,;:")
        if not skill:
            continue
        key = normalize(skill)
        if key not in seen:
            result.append(skill)
            seen.add(key)
    return result

def _skill_terms(skill: str) -> list[str]:
    return [skill] + SKILL_ALIASES.get(normalize(skill), [])

def _is_negated(line: str, term: str) -> bool:
    normalized_line, normalized_term = normalize(line), normalize(term)
    match = re.search(rf"(?<!\w){re.escape(normalized_term)}(?!\w)", normalized_line)
    if not match:
        return False
    prefix = normalized_line[max(0, match.start() - 65):match.start()]
    prefix = re.split(r"[.;]", prefix)[-1]
    return bool(NEGATION_PATTERN.search(prefix))

def skill_in_text(text: str, skill: str) -> bool:
    for line in text.splitlines():
        for term in _skill_terms(skill):
            if contains_term(line, term) and not _is_negated(line, term):
                return True
    return False

def find_skill_evidence(text: str, skill: str, width: int = 180) -> str:
    for line in text.splitlines():
        line = line.strip()
        for term in _skill_terms(skill):
            if contains_term(line, term):
                if _is_negated(line, term):
                    continue
                if len(line) <= width:
                    return line
                normalized_term = normalize(term)
                words = line.split()
                for index, word in enumerate(words):
                    if normalize(word) in normalized_term.split():
                        start = max(0, index - 12)
                        end = min(len(words), index + 18)
                        return " ".join(words[start:end])
                return line[:width]
    return ""

SECTION_NAMES = {
    "experience": ["experience", "work experience", "professional experience", "employment history", "work history", "career history"],
    "education": ["education", "academic background", "academic qualifications", "educational qualifications"],
    "projects": ["projects", "personal projects", "academic projects", "key projects", "relevant projects"],
    "skills": ["skills", "technical skills", "core skills", "key skills"],
}

OTHER_HEADINGS = {
    "summary", "professional summary", "profile", "certifications",
    "certificates", "languages", "references", "achievements",
    "awards", "publications", "interests", "objective",
}

ALL_HEADINGS = set(OTHER_HEADINGS)
for names in SECTION_NAMES.values():
    ALL_HEADINGS.update(names)

def _heading(line: str) -> str:
    return re.sub(r"[^a-z ]", "", line.lower()).strip()

def extract_section(text: str, section_name: str) -> str:
    headings = SECTION_NAMES.get(section_name, [])
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if _heading(line) in headings:
            start = index + 1
            break
    if start is None:
        return ""
    result = []
    for line in lines[start:]:
        if _heading(line) in ALL_HEADINGS:
            break
        result.append(line)
    return "\n".join(result).strip()

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

DATE_TOKEN = r"(?:[A-Za-z]{3,9}\.?\s+\d{4}|\d{1,2}/\d{4})"
DATE_RANGE = re.compile(
    rf"({DATE_TOKEN})\s*(?:-|–|—|to)\s*({DATE_TOKEN}|present|current|currently|now)",
    re.IGNORECASE,
)

def _month_index(token: str) -> Optional[int]:
    token = token.lower().strip().replace(".", "")
    today = date.today()
    if token in {"present", "current", "currently", "now"}:
        return today.year * 12 + today.month
    match = re.match(r"(\d{1,2})/(\d{4})", token)
    if match:
        month, year = int(match.group(1)), int(match.group(2))
        return year * 12 + month if 1 <= month <= 12 else None
    match = re.match(r"([a-z]+)\s+(\d{4})", token)
    if match:
        month = MONTHS.get(match.group(1)[:3])
        year = int(match.group(2))
        if month:
            return year * 12 + month
    return None

def _merge_intervals(intervals: list[tuple[int, int]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        previous = merged[-1]
        if start <= previous[1] + 1:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])
    months = sum(end - start + 1 for start, end in merged)
    return round(months / 12, 2)

def extract_experience(text: str) -> Optional[float]:
    section = extract_section(text, "experience")
    intervals = []
    for match in DATE_RANGE.finditer(section):
        start, end = _month_index(match.group(1)), _month_index(match.group(2))
        if start is None or end is None or end < start:
            continue
        intervals.append((start, end))
    if intervals:
        return _merge_intervals(intervals)
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience",
        r"experience\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*\+?\s*years?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None

def analyze_experience(text: str) -> tuple[Optional[float], str]:
    years = extract_experience(text)
    if years is None:
        return None, "Experience duration could not be reliably determined."
    return years, (
        f"{years:g} years of documented experience. "
        "This is total experience, not necessarily experience relevant to the requested position."
    )

ROLE_KEYWORDS = {
    "accountant": [
        "accountant", "accounting", "accounts assistant", "bookkeeper",
        "bookkeeping", "finance officer", "financial accountant", "audit assistant",
    ],
    "data scientist": [
        "data scientist", "data science", "machine learning engineer", "ml engineer",
    ],
    "ai engineer": [
        "ai engineer", "artificial intelligence engineer", "machine learning engineer", "ml engineer",
    ],
    "software engineer": [
        "software engineer", "software developer", "backend developer",
        "frontend developer", "full stack developer",
    ],
}

def _role_terms(job_title: str) -> list[str]:
    title = normalize(job_title)
    return ROLE_KEYWORDS.get(title, [title] if title else [])

def extract_relevant_experience(text: str, job_title: str) -> Optional[float]:
    section = extract_section(text, "experience")
    if not section:
        return None
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    terms = _role_terms(job_title)
    if not terms:
        return None
    intervals = []
    for index, line in enumerate(lines):
        match = DATE_RANGE.search(line)
        if not match:
            continue
        start, end = _month_index(match.group(1)), _month_index(match.group(2))
        if start is None or end is None or end < start:
            continue
        nearby = " ".join(lines[max(0, index - 2):index + 2])
        if any(contains_term(nearby, term) for term in terms):
            intervals.append((start, end))
    return _merge_intervals(intervals) if intervals else None

DEGREE_PATTERN = re.compile(
    r"\b(bachelor(?:'s)?|b\.?\s?sc\.?|b\.?\s?com\.?|b\.?\s?ba\.?|bba|btech|b\.?\s?s\.?|master(?:'s)?|mba|msc|m\.?\s?s\.?|phd|doctorate|diploma)\b",
    re.IGNORECASE,
)

def extract_education(text: str) -> list[str]:
    section = extract_section(text, "education")
    source = section if section else text
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    results, seen = [], set()
    for index, line in enumerate(lines):
        if not DEGREE_PATTERN.search(line):
            continue
        if not section and re.search(
            r"\b(graduate with|experience across|experienced in)\b",
            line, re.IGNORECASE
        ):
            continue
        snippet = line
        if index + 1 < len(lines):
            next_line = lines[index + 1]
            if re.search(r"\b(university|college|institute)\b", next_line, re.IGNORECASE):
                snippet += " | " + next_line
        key = normalize(snippet)
        if key not in seen:
            results.append(snippet)
            seen.add(key)
    return results

FIELD_ALIASES = {
    "accounting": ["accounting", "accountancy", "bcom", "b com", "commerce"],
    "finance": ["finance", "financial management"],
    "commerce": ["commerce", "bcom", "b com"],
    "business administration": ["business administration", "bba", "mba"],
    "computer science": ["computer science", "bscs", "bs cs"],
    "artificial intelligence": ["artificial intelligence", "bs ai"],
}

def qualification_matches(education_text: str, qualification: str) -> bool:
    education, requirement = normalize(education_text), normalize(qualification)
    if not education or not requirement:
        return False
    if contains_term(education, requirement):
        return True
    for field, aliases in FIELD_ALIASES.items():
        required_aliases = [field] + aliases
        if any(contains_term(requirement, alias) for alias in required_aliases):
            if any(contains_term(education, alias) for alias in required_aliases):
                return True
    return False

def parse_qualifications(text: str) -> list[str]:
    return [
        item.strip().rstrip(".")
        for item in re.split(r"[,;\n]", text)
        if item.strip()
    ]

STOPWORDS = {
    "the", "and", "for", "with", "are", "this", "that", "have",
    "will", "from", "into", "their", "they", "our", "you", "your",
    "should", "must", "can", "may", "all", "use", "using",
    "able", "work", "working", "required", "candidate",
    "ideal", "responsibilities", "seeking",
}

def jd_overlap_score(resume_text: str, job_description: str) -> float:
    jd_terms = {
        word for word in normalize(job_description).split()
        if len(word) > 3 and word not in STOPWORDS
    }
    resume_terms = set(normalize(resume_text).split())
    if not jd_terms:
        return 0.0
    return round(100 * len(jd_terms & resume_terms) / len(jd_terms), 2)

WEIGHTS = {
    "skills": 0.40,
    "experience": 0.25,
    "qualifications": 0.15,
    "description": 0.20,
}

def score_resume(
    resume_text: str,
    required_skills: list[str],
    min_years: float,
    max_years: float,
    qualifications: str,
    job_description: str,
    analysis: dict | None = None,
    job_title: str = "",
) -> dict:
    if min_years > max_years:
        raise ValueError("Minimum experience cannot exceed maximum experience.")

    analysis = analysis or {}
    skills = clean_skills(required_skills)
    matched_skills = [skill for skill in skills if skill_in_text(resume_text, skill)]
    missing_skills = [skill for skill in skills if skill not in matched_skills]
    skill_score = 100 * len(matched_skills) / len(skills) if skills else 0
    skill_evidence = {
        skill: find_skill_evidence(resume_text, skill)
        for skill in matched_skills
    }

    total_experience, experience_summary = analyze_experience(resume_text)
    relevant_experience = extract_relevant_experience(resume_text, job_title)

    if relevant_experience is None:
        experience_score = 0.0
        experience_status = "Relevant dated experience not verified"
    elif relevant_experience < min_years:
        experience_score = (
            100 * relevant_experience / min_years if min_years > 0 else 100
        )
        experience_status = "Below minimum relevant experience"
    elif relevant_experience > max_years:
        experience_score = 100.0
        experience_status = "Meets minimum; above preferred maximum"
    else:
        experience_score = 100.0
        experience_status = "Within requested range"

    education_snippets = extract_education(resume_text)
    education_text = "\n".join(education_snippets)
    qualification_terms = parse_qualifications(qualifications)

    matched_qualifications = [
        qualification for qualification in qualification_terms
        if qualification_matches(education_text, qualification)
    ]

    qualification_score = 100.0 if matched_qualifications else 0.0
    description_score = jd_overlap_score(resume_text, job_description)

    component_scores = {
        "skills": skill_score,
        "experience": experience_score,
        "qualifications": qualification_score,
        "description": description_score,
    }

    active_weights = dict(WEIGHTS)
    if not skills:
        active_weights.pop("skills")
    if not qualification_terms:
        active_weights.pop("qualifications")
    if not job_description.strip():
        active_weights.pop("description")

    weight_total = sum(active_weights.values())
    final_score = (
        sum(component_scores[key] * weight for key, weight in active_weights.items()) / weight_total
        if weight_total else 0
    )

    matching_evidence = [
        f"{skill}: {evidence}"
        for skill, evidence in skill_evidence.items()
        if evidence
    ]

    missing_requirements = [
        f"{skill}: no positive evidence identified in resume."
        for skill in missing_skills
    ]

    if relevant_experience is None:
        missing_requirements.append("Relevant dated experience could not be verified.")

    if qualification_terms and not matched_qualifications:
        missing_requirements.append("No accepted qualification was identified.")

    merged_analysis = {
        "relevant_projects": analysis.get("relevant_projects", []),
        "experience_summary": experience_summary,
        "qualifications": [
            {"name": "Education", "evidence": item}
            for item in education_snippets
        ],
        "matching_evidence": matching_evidence,
        "missing_requirements": missing_requirements,
    }

    return {
        "score": round(final_score, 2),
        "skill_score": round(skill_score, 2),
        "experience_score": round(experience_score, 2),
        "qualification_score": round(qualification_score, 2),
        "description_score": description_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_qualifications": matched_qualifications,
        "experience_years": total_experience,
        "relevant_experience_years": relevant_experience,
        "experience_status": experience_status,
        "education_evidence": education_snippets,
        "skill_evidence": skill_evidence,
        "analysis": merged_analysis,
    }

def rank_resumes(candidates: list[dict]) -> list[dict]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (candidate["score"], candidate["skill_score"]),
        reverse=True,
    )
    return [
        {**candidate, "rank": index}
        for index, candidate in enumerate(ranked, start=1)
    ]

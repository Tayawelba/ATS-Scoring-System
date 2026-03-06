import html
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pdfplumber
import spacy
import streamlit as st
from docx import Document
from spacy.lang.en.stop_words import STOP_WORDS


MODEL_PATH = Path(__file__).resolve().parent / "ner_model"

SECTION_PATTERNS = {
    "summary": [r"\bsummary\b", r"\bprofessional profile\b", r"\bprofile\b", r"\bobjective\b"],
    "experience": [r"\bwork experience\b", r"\bexperience\b", r"\bemployment\b", r"\bprofessional experience\b"],
    "education": [r"\beducation\b", r"\bacademic\b"],
    "skills": [r"\bskills\b", r"\btechnical skills\b", r"\bcore competencies\b", r"\btech stack\b"],
    "projects": [r"\bprojects\b", r"\bselected projects\b", r"\bproject experience\b"],
    "certifications": [r"\bcertifications\b", r"\blicenses\b", r"\bcertificates\b"],
}

JOB_SIGNAL_PATTERNS = [
    r"(?:must have|required|required skills|requirements|qualifications|skills|proficient in|experience with|hands-on experience with|knowledge of|expertise in|strong (?:experience|knowledge) of)\s*:?\s*([^\n.;]+)",
]

GENERIC_TERMS = {
    "ability",
    "abilities",
    "additional",
    "applicant",
    "candidate",
    "company",
    "communication",
    "detail",
    "driven",
    "education",
    "excellent",
    "experience",
    "experienced",
    "expertise",
    "good",
    "great",
    "ideal",
    "including",
    "information",
    "job",
    "knowledge",
    "looking",
    "must",
    "preferred",
    "present",
    "professional",
    "profile",
    "qualification",
    "qualifications",
    "requirement",
    "requirements",
    "responsibilities",
    "resume",
    "role",
    "skills",
    "strong",
    "summary",
    "team",
    "tools",
    "using",
    "various",
    "vitae",
    "work",
    "worked",
    "year",
    "years",
}

SECTION_TITLES = {
    "work experience",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "additional information",
    "professional skills",
}

ALIASES = {
    "ai": "artificial intelligence",
    "aws": "amazon web services",
    "azure sql": "sql azure",
    "c sharp": "c#",
    "ci cd": "ci/cd",
    "gcp": "google cloud platform",
    "genai": "generative ai",
    "js": "javascript",
    "llm": "large language model",
    "llms": "large language model",
    "machine-learning": "machine learning",
    "ml": "machine learning",
    "mongo": "mongodb",
    "ms excel": "excel",
    "microsoft excel": "excel",
    "nlp": "natural language processing",
    "node js": "nodejs",
    "node.js": "nodejs",
    "postgres": "postgresql",
    "powerbi": "power bi",
    "react js": "react",
    "react.js": "react",
    "ts": "typescript",
    "u sql": "u-sql",
    "vue js": "vue",
}

DEGREE_PATTERNS = {
    "doctorate": [r"\bphd\b", r"\bdoctorate\b", r"\bdoctoral\b"],
    "master": [r"\bmaster'?s\b", r"\bmba\b", r"\bm\.?sc\b", r"\bm\.?s\b"],
    "bachelor": [r"\bbachelor'?s\b", r"\bb\.?sc\b", r"\bb\.?s\b", r"\bb\.?a\b"],
    "associate": [r"\bassociate'?s\b"],
    "diploma": [r"\bdiploma\b", r"\bcertificate\b"],
}

DEGREE_RANK = {
    "diploma": 1,
    "associate": 2,
    "bachelor": 3,
    "master": 4,
    "doctorate": 5,
}

CONTACT_PATTERNS = {
    "email": re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE),
    "phone": re.compile(r"(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,}\d{3,4})"),
}


@st.cache_resource(show_spinner=False)
def load_nlp():
    return spacy.load(MODEL_PATH)


def extract_text_from_pdf(pdf_file):
    pdf_file.seek(0)
    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def extract_text_from_txt(text_file):
    text_file.seek(0)
    raw = text_file.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    return raw


def extract_text_from_docx(word_file):
    word_file.seek(0)
    document = Document(word_file)
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text)


def get_uploaded_file_text(uploaded_resume):
    content_type = uploaded_resume.type or ""
    file_name = uploaded_resume.name.lower()

    if content_type == "application/pdf" or file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_resume)
    if content_type in {"text/plain", "text/markdown"} or file_name.endswith(".txt"):
        return extract_text_from_txt(uploaded_resume)
    if "wordprocessingml" in content_type or file_name.endswith(".docx"):
        return extract_text_from_docx(uploaded_resume)
    raise ValueError("Le fichier doit etre au format PDF, DOCX ou TXT.")


def preprocess_text(text):
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2022": "-",
        "\uf0b7": "-",
        "\xa0": " ",
        "â€“": "-",
        "â€”": "-",
        "â€¢": "-",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€¦": "...",
    }

    text = html.unescape(text or "")
    text = unicodedata.normalize("NFKC", text)
    for wrong_value, correct_value in replacements.items():
        text = text.replace(wrong_value, correct_value)

    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_token(token):
    token = token.strip().lower()
    token = token.replace("&", " and ")
    token = token.replace(".", "")
    token = token.replace("/", " ")
    token = re.sub(r"[^a-z0-9+#\s-]+", " ", token)
    token = re.sub(r"\s+", " ", token).strip()

    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        token = token[:-1]

    return ALIASES.get(token, token)


def normalize_phrase(text):
    words = []
    raw_words = text.split()
    for raw_word in raw_words:
        normalized_word = normalize_token(raw_word)
        if not normalized_word:
            continue
        if normalized_word in STOP_WORDS and len(raw_words) > 1:
            continue
        words.append(normalized_word)

    phrase = " ".join(words).strip()
    phrase = ALIASES.get(phrase, phrase)
    return re.sub(r"\s+", " ", phrase)


def dedupe(items):
    seen = set()
    ordered_items = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered_items.append(item)
    return ordered_items


def is_valid_phrase(phrase):
    if not phrase or phrase in SECTION_TITLES:
        return False

    words = phrase.split()
    if len(words) > 5:
        return False
    if len(words) == 1 and (len(words[0]) < 2 or words[0] in STOP_WORDS):
        return False
    if all(word in GENERIC_TERMS for word in words):
        return False
    if re.fullmatch(r"[\d\s.+#-]+", phrase):
        return False
    return True


def looks_like_skill_phrase(phrase):
    if not is_valid_phrase(phrase):
        return False

    words = phrase.split()
    if len(words) > 4:
        return False
    return any(word not in GENERIC_TERMS for word in words)


def split_list_items(fragment):
    cleaned_fragment = fragment.lower()
    cleaned_fragment = re.sub(
        r"^(?:experience in|experience with|knowledge of|proficient in|expertise in|worked on|hands-on experience with|strong knowledge of)\s+",
        "",
        cleaned_fragment,
    )
    cleaned_fragment = cleaned_fragment.replace("|", ",")
    cleaned_fragment = cleaned_fragment.replace(" + ", ", ")
    cleaned_fragment = cleaned_fragment.replace(" and ", ", ")
    cleaned_fragment = cleaned_fragment.replace(" or ", ", ")

    normalized_items = []
    for item in re.split(r"[,;/]", cleaned_fragment):
        normalized_item = normalize_phrase(item)
        if is_valid_phrase(normalized_item):
            normalized_items.append(normalized_item)
    return normalized_items


def extract_keyword_tokens(text):
    keyword_counts = Counter()
    for raw_token in re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{1,}", text):
        normalized_token = normalize_token(raw_token)
        if not normalized_token:
            continue
        if normalized_token in STOP_WORDS or normalized_token in GENERIC_TERMS:
            continue
        if len(normalized_token) < 2:
            continue
        keyword_counts[normalized_token] += 1
    return keyword_counts


def extract_phrase_candidates(text):
    candidates = []
    for line in text.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue
        for fragment in re.split(r"[|,;]", stripped_line):
            candidates.extend(split_list_items(fragment))
    return dedupe(candidates)


def extract_entities(text):
    entity_rows = []
    grouped_entities = defaultdict(set)
    doc = load_nlp()(text)
    for entity in doc.ents:
        value = entity.text.strip(" \n\t,;:-")
        if not value:
            continue
        entity_rows.append({"Label": entity.label_, "Entity": value})
        grouped_entities[entity.label_].add(value)

    return entity_rows, {label: sorted(values) for label, values in grouped_entities.items()}


def detect_sections(text):
    lowered_text = text.lower()
    return {
        section_name: any(re.search(pattern, lowered_text) for pattern in patterns)
        for section_name, patterns in SECTION_PATTERNS.items()
    }


def detect_contact_details(text):
    lowered_text = text.lower()
    return {
        "email": bool(CONTACT_PATTERNS["email"].search(text)),
        "phone": bool(CONTACT_PATTERNS["phone"].search(text)),
        "linkedin": "linkedin.com/" in lowered_text or "linkedin" in lowered_text,
        "portfolio": any(
            domain in lowered_text
            for domain in ("github.com/", "gitlab.com/", "behance.net/", "dribbble.com/", "portfolio")
        ),
    }


def has_quantified_achievements(text):
    lowered_text = text.lower()
    return bool(
        re.search(r"\b\d+(?:\.\d+)?\s*(?:%|percent|k|m|b|million|billion|\+)\b", lowered_text)
        or re.search(r"\b(increased|reduced|grew|improved|saved|delivered)\b", lowered_text)
    )


def parse_year_values(text):
    lowered_text = text.lower()
    ranges = [(int(low), int(high)) for low, high in re.findall(r"(\d+)\s*-\s*(\d+)\s*(?:years?|yrs?)", lowered_text)]
    singles = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", lowered_text)]
    return ranges, singles


def extract_resume_years(text, entity_map):
    year_values = []
    for value in entity_map.get("YEARS_OF_EXPERIENCE", []):
        _, singles = parse_year_values(value)
        year_values.extend(singles)

    ranges, singles = parse_year_values(text)
    year_values.extend(singles)
    year_values.extend(high for _, high in ranges)

    return max(year_values) if year_values else None


def extract_required_years(text, entity_map):
    requirement_values = []
    for value in entity_map.get("YEARS_OF_EXPERIENCE", []):
        ranges, singles = parse_year_values(value)
        requirement_values.extend(low for low, _ in ranges)
        requirement_values.extend(singles)

    ranges, singles = parse_year_values(text)
    requirement_values.extend(low for low, _ in ranges)
    requirement_values.extend(singles)

    return min(requirement_values) if requirement_values else None


def extract_degree_levels(text, entity_map):
    levels = set()
    lowered_text = text.lower()
    for level, patterns in DEGREE_PATTERNS.items():
        if any(re.search(pattern, lowered_text) for pattern in patterns):
            levels.add(level)

    for degree_value in entity_map.get("DEGREE", []):
        lowered_value = degree_value.lower()
        for level, patterns in DEGREE_PATTERNS.items():
            if any(re.search(pattern, lowered_value) for pattern in patterns):
                levels.add(level)
    return levels


def extract_job_requirement_fragments(text):
    fragments = []
    for pattern in JOB_SIGNAL_PATTERNS:
        fragments.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return fragments


def extract_job_skills(text, entity_map, phrase_candidates):
    skill_terms = set()

    for label in ("SKILLS", "DESIGNATION"):
        for value in entity_map.get(label, []):
            skill_terms.update(split_list_items(value))

    for fragment in extract_job_requirement_fragments(text):
        skill_terms.update(split_list_items(fragment))

    for line in text.splitlines():
        lowered_line = line.lower()
        if " with " in lowered_line:
            skill_terms.update(split_list_items(lowered_line.split(" with ", 1)[1]))

    skill_terms.update(phrase for phrase in phrase_candidates if looks_like_skill_phrase(phrase))

    return sorted(skill_terms)


def extract_resume_skills(text, entity_map, phrase_candidates):
    skill_terms = set()

    for value in entity_map.get("SKILLS", []):
        skill_terms.update(split_list_items(value))

    for phrase in phrase_candidates:
        if looks_like_skill_phrase(phrase):
            skill_terms.add(phrase)

    for line in text.splitlines():
        lowered_line = line.lower()
        if any(keyword in lowered_line for keyword in ("skills", "tools", "technologies", "stack", "frameworks")):
            skill_terms.update(split_list_items(line))

    return sorted(skill_terms)


def build_resume_profile(text):
    entity_rows, entity_map = extract_entities(text)
    keyword_counts = extract_keyword_tokens(text)
    phrase_candidates = extract_phrase_candidates(text)

    top_keywords = [token for token, _ in keyword_counts.most_common(25)]
    resume_skills = extract_resume_skills(text, entity_map, phrase_candidates)
    designations = dedupe(
        normalize_phrase(value)
        for value in entity_map.get("DESIGNATION", [])
        if is_valid_phrase(normalize_phrase(value))
    )

    candidate_terms = set(resume_skills)
    candidate_terms.update(phrase_candidates)
    candidate_terms.update(top_keywords)
    candidate_terms.update(designations)

    return {
        "entity_rows": entity_rows,
        "entity_map": entity_map,
        "keyword_counts": keyword_counts,
        "top_keywords": top_keywords,
        "phrase_candidates": phrase_candidates,
        "skills": resume_skills,
        "designations": designations,
        "degree_levels": extract_degree_levels(text, entity_map),
        "years_of_experience": extract_resume_years(text, entity_map),
        "sections": detect_sections(text),
        "contacts": detect_contact_details(text),
        "has_quantified_achievements": has_quantified_achievements(text),
        "word_count": len(text.split()),
        "candidate_terms": candidate_terms,
    }


def build_job_profile(text):
    entity_rows, entity_map = extract_entities(text)
    keyword_counts = extract_keyword_tokens(text)
    phrase_candidates = extract_phrase_candidates(text)

    top_keywords = [token for token, _ in keyword_counts.most_common(20)]
    required_skills = extract_job_skills(text, entity_map, phrase_candidates)
    candidate_terms = set(required_skills)
    candidate_terms.update(top_keywords)
    candidate_terms.update(phrase_candidates)

    return {
        "entity_rows": entity_rows,
        "entity_map": entity_map,
        "keyword_counts": keyword_counts,
        "top_keywords": top_keywords,
        "phrase_candidates": phrase_candidates,
        "skills": required_skills,
        "degree_levels": extract_degree_levels(text, entity_map),
        "required_years": extract_required_years(text, entity_map),
        "candidate_terms": candidate_terms,
    }


def token_set(value):
    normalized_tokens = set()
    for raw_token in re.findall(r"[A-Za-z0-9+#./-]+", value):
        normalized = normalize_token(raw_token)
        if normalized:
            normalized_tokens.add(normalized)
    return normalized_tokens


def find_term_match(required_term, available_terms):
    required_tokens = token_set(required_term)
    if not required_tokens:
        return None

    for available_term in available_terms:
        if required_term == available_term:
            return available_term

        available_tokens = token_set(available_term)
        if not available_tokens:
            continue

        overlap = len(required_tokens & available_tokens)
        if required_tokens <= available_tokens or available_tokens <= required_tokens:
            return available_term
        if overlap / max(len(required_tokens), len(available_tokens)) >= 0.75:
            return available_term
    return None


def score_skill_alignment(resume_profile, job_profile):
    required_skills = job_profile["skills"]
    if not required_skills:
        return None

    matched_skills = {}
    for skill in required_skills:
        matched_value = find_term_match(skill, resume_profile["candidate_terms"])
        if matched_value:
            matched_skills[skill] = matched_value

    missing_skills = [skill for skill in required_skills if skill not in matched_skills]
    score = round((len(matched_skills) / len(required_skills)) * 100)

    return {
        "name": "Skills coverage",
        "score": score,
        "weight": 40,
        "summary": f"{len(matched_skills)}/{len(required_skills)} required skills covered",
        "matched_terms": sorted(matched_skills),
        "missing_terms": missing_skills,
    }


def score_keyword_alignment(resume_profile, job_profile):
    job_keywords = job_profile["top_keywords"]
    if not job_keywords:
        return None

    resume_terms = set(resume_profile["keyword_counts"]) | resume_profile["candidate_terms"]
    matched_keywords = [keyword for keyword in job_keywords if keyword in resume_terms]
    missing_keywords = [keyword for keyword in job_keywords if keyword not in resume_terms]
    score = round((len(matched_keywords) / len(job_keywords)) * 100)

    return {
        "name": "Keyword coverage",
        "score": score,
        "weight": 20,
        "summary": f"{len(matched_keywords)}/{len(job_keywords)} important job keywords found in the resume",
        "matched_terms": matched_keywords,
        "missing_terms": missing_keywords,
    }


def score_experience_alignment(resume_profile, job_profile):
    required_years = job_profile["required_years"]
    if required_years is None:
        return None

    resume_years = resume_profile["years_of_experience"]
    if resume_years is None:
        return {
            "name": "Experience alignment",
            "score": 20 if resume_profile["sections"]["experience"] else 0,
            "weight": 20,
            "summary": f"Job asks for about {required_years:g}+ years, but the resume does not state a clear total",
            "matched_terms": [],
            "missing_terms": [f"{required_years:g}+ years of experience"],
        }

    score = round(min(resume_years / required_years, 1.0) * 100)
    return {
        "name": "Experience alignment",
        "score": score,
        "weight": 20,
        "summary": f"Resume shows {resume_years:g} years for a {required_years:g}+ year requirement",
        "matched_terms": [f"{resume_years:g} years"] if resume_years >= required_years else [],
        "missing_terms": [] if resume_years >= required_years else [f"{required_years:g}+ years"],
    }


def score_education_alignment(resume_profile, job_profile):
    required_degrees = job_profile["degree_levels"]
    if not required_degrees:
        return None

    required_rank = max(DEGREE_RANK[level] for level in required_degrees)
    resume_ranks = [DEGREE_RANK[level] for level in resume_profile["degree_levels"]]
    resume_rank = max(resume_ranks) if resume_ranks else 0

    if resume_rank == 0:
        score = 0
    else:
        score = round(min(resume_rank / required_rank, 1.0) * 100)

    return {
        "name": "Education alignment",
        "score": score,
        "weight": 10,
        "summary": f"Job requires {', '.join(sorted(required_degrees))}; resume shows {', '.join(sorted(resume_profile['degree_levels'])) or 'no clear degree'}",
        "matched_terms": sorted(resume_profile["degree_levels"] & required_degrees),
        "missing_terms": [] if resume_rank >= required_rank else sorted(required_degrees),
    }


def score_resume_quality(resume_profile):
    score = 0
    strengths = []
    recommendations = []
    sections = resume_profile["sections"]
    contacts = resume_profile["contacts"]
    word_count = resume_profile["word_count"]

    if contacts["email"]:
        score += 10
    else:
        recommendations.append("Add a professional email address.")

    if contacts["phone"]:
        score += 8
    else:
        recommendations.append("Add a phone number so recruiters can reach you quickly.")

    if contacts["linkedin"] or contacts["portfolio"]:
        score += 4
    else:
        recommendations.append("Add a LinkedIn, GitHub or portfolio link.")

    if sections["summary"]:
        score += 10
    else:
        recommendations.append("Add a short professional summary tailored to the target role.")

    if sections["experience"]:
        score += 18
        strengths.append("Work experience section detected.")
    else:
        recommendations.append("Add a clear work experience section.")

    if sections["education"]:
        score += 12
    else:
        recommendations.append("Add an education section.")

    if sections["skills"]:
        score += 12
        strengths.append("Skills section detected.")
    else:
        recommendations.append("Add a dedicated skills section.")

    if sections["projects"] or sections["certifications"]:
        score += 8
    else:
        recommendations.append("Highlight projects or certifications relevant to the role.")

    if resume_profile["has_quantified_achievements"]:
        score += 8
        strengths.append("Quantified achievements detected.")
    else:
        recommendations.append("Add measurable results such as percentages, revenue, time saved or scale.")

    if 250 <= word_count <= 900:
        score += 18
    elif 150 <= word_count < 250 or 900 < word_count <= 1200:
        score += 10
    else:
        recommendations.append("Keep the resume concise and information-dense, ideally between 250 and 900 words.")

    return {
        "name": "Resume quality",
        "score": score,
        "weight": 10,
        "summary": "Resume quality score based on structure, contact details and evidence",
        "matched_terms": [],
        "missing_terms": [],
        "strengths": dedupe(strengths),
        "recommendations": dedupe(recommendations),
    }


def combine_dimensions(dimensions):
    active_dimensions = [dimension for dimension in dimensions if dimension is not None]
    total_weight = sum(dimension["weight"] for dimension in active_dimensions)
    overall_score = round(sum(dimension["score"] * dimension["weight"] for dimension in active_dimensions) / total_weight)

    normalized_dimensions = []
    for dimension in active_dimensions:
        normalized_dimension = dict(dimension)
        normalized_dimension["effective_weight"] = round((dimension["weight"] / total_weight) * 100, 1)
        normalized_dimensions.append(normalized_dimension)
    return overall_score, normalized_dimensions


def dedupe_messages(messages):
    return dedupe(message.strip() for message in messages if message and message.strip())


def rate_score(score):
    if score >= 85:
        return "Tres bon"
    if score >= 70:
        return "Bon"
    if score >= 55:
        return "Moyen"
    return "Faible"


def format_terms(terms, limit=6):
    limited_terms = terms[:limit]
    if not limited_terms:
        return "None"
    suffix = "" if len(terms) <= limit else ", ..."
    return ", ".join(limited_terms) + suffix


def build_overall_summary(overall_score, skill_dimension, experience_dimension, quality_dimension):
    summary_parts = [f"Score global {overall_score}/100 ({rate_score(overall_score).lower()})."]
    if skill_dimension:
        summary_parts.append(skill_dimension["summary"] + ".")
    if experience_dimension:
        summary_parts.append(experience_dimension["summary"] + ".")
    summary_parts.append(f"Resume quality: {quality_dimension['score']}/100.")
    return " ".join(summary_parts)


def evaluate_resume_against_job(resume_text, job_text):
    cleaned_resume = preprocess_text(resume_text)
    cleaned_job = preprocess_text(job_text)

    resume_profile = build_resume_profile(cleaned_resume)
    job_profile = build_job_profile(cleaned_job)

    skill_dimension = score_skill_alignment(resume_profile, job_profile)
    keyword_dimension = score_keyword_alignment(resume_profile, job_profile)
    experience_dimension = score_experience_alignment(resume_profile, job_profile)
    education_dimension = score_education_alignment(resume_profile, job_profile)
    quality_dimension = score_resume_quality(resume_profile)

    overall_score, dimensions = combine_dimensions(
        [skill_dimension, keyword_dimension, experience_dimension, education_dimension, quality_dimension]
    )

    strengths = list(quality_dimension["strengths"])
    recommendations = list(quality_dimension["recommendations"])

    if skill_dimension and skill_dimension["matched_terms"]:
        strengths.append(f"Relevant skills already present: {format_terms(skill_dimension['matched_terms'])}.")
    if skill_dimension and skill_dimension["missing_terms"]:
        recommendations.append(f"Add or make clearer these required skills: {format_terms(skill_dimension['missing_terms'])}.")

    if keyword_dimension and keyword_dimension["missing_terms"]:
        recommendations.append(f"Use more of the job language in the resume: {format_terms(keyword_dimension['missing_terms'])}.")

    if experience_dimension and experience_dimension["score"] >= 100:
        strengths.append("Experience appears to meet the stated requirement.")
    elif experience_dimension and experience_dimension["missing_terms"]:
        recommendations.append("State your total years of experience more clearly and connect them to the target role.")

    if education_dimension and education_dimension["score"] >= 100:
        strengths.append("Education appears aligned with the role.")
    elif education_dimension and education_dimension["missing_terms"]:
        recommendations.append("Make the relevant degree or training easier to spot.")

    strengths = dedupe_messages(strengths)
    recommendations = dedupe_messages(recommendations)

    if not strengths:
        strengths = ["The resume has usable content, but it needs stronger alignment with the target role."]

    return {
        "overall_score": overall_score,
        "rating": rate_score(overall_score),
        "summary": build_overall_summary(overall_score, skill_dimension, experience_dimension, quality_dimension),
        "dimensions": dimensions,
        "strengths": strengths,
        "recommendations": recommendations,
        "resume_profile": resume_profile,
        "job_profile": job_profile,
    }


def evaluate_resume_only(resume_text):
    cleaned_resume = preprocess_text(resume_text)
    resume_profile = build_resume_profile(cleaned_resume)
    quality_dimension = score_resume_quality(resume_profile)

    strengths = list(quality_dimension["strengths"])
    recommendations = list(quality_dimension["recommendations"])

    if resume_profile["skills"]:
        strengths.append(f"Detected skill signals: {format_terms(resume_profile['skills'])}.")
    if resume_profile["years_of_experience"] is not None:
        strengths.append(f"Detected experience signal: about {resume_profile['years_of_experience']:g} years.")

    strengths = dedupe_messages(strengths)
    recommendations = dedupe_messages(recommendations)

    if not strengths:
        strengths = ["The resume can be scored structurally, but it still needs richer evidence and clearer sections."]

    return {
        "overall_score": quality_dimension["score"],
        "rating": rate_score(quality_dimension["score"]),
        "summary": f"Resume-only quality score: {quality_dimension['score']}/100.",
        "dimensions": [dict(quality_dimension, effective_weight=100.0)],
        "strengths": strengths,
        "recommendations": recommendations,
        "resume_profile": resume_profile,
        "job_profile": None,
    }


def render_dimension_table(dimensions):
    table = [
        {
            "Dimension": dimension["name"],
            "Score": dimension["score"],
            "Weight (%)": dimension["effective_weight"],
            "Comment": dimension["summary"],
        }
        for dimension in dimensions
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_list(title, items, empty_message):
    st.write(f"### {title}")
    if items:
        for item in items:
            st.write(f"- {item}")
    else:
        st.write(empty_message)


def render_entities_section(resume_profile, job_profile=None):
    with st.expander("Voir les entites detectees"):
        resume_entities = resume_profile["entity_rows"]
        if resume_entities:
            st.write("#### CV")
            st.dataframe(resume_entities, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune entite n'a ete detectee dans le CV.")

        if job_profile is not None:
            job_entities = job_profile["entity_rows"]
            if job_entities:
                st.write("#### Offre")
                st.dataframe(job_entities, use_container_width=True, hide_index=True)
            else:
                st.info("Aucune entite n'a ete detectee dans l'offre.")


def render_results(evaluation, has_job_description):
    score_column, rating_column, detail_column = st.columns(3)
    score_column.metric("Score global", f"{evaluation['overall_score']}/100")
    rating_column.metric("Niveau", evaluation["rating"])

    if has_job_description and evaluation["job_profile"] is not None:
        matched_skills = next(
            (dimension for dimension in evaluation["dimensions"] if dimension["name"] == "Skills coverage"),
            None,
        )
        if matched_skills is not None:
            detail_column.metric(
                "Competences couvertes",
                f"{len(matched_skills['matched_terms'])}/{len(evaluation['job_profile']['skills']) or 0}",
            )
        else:
            detail_column.metric("Competences couvertes", "N/A")
    else:
        detail_column.metric("Mots detectes", str(len(evaluation["resume_profile"]["top_keywords"])))

    st.progress(evaluation["overall_score"] / 100)
    st.write(evaluation["summary"])

    st.write("## Analyse detaillee")
    render_dimension_table(evaluation["dimensions"])

    render_list("Points forts", evaluation["strengths"], "Aucun point fort clair n'a ete detecte.")
    render_list("Actions recommandees", evaluation["recommendations"], "Aucune action prioritaire.")

    render_entities_section(evaluation["resume_profile"], evaluation["job_profile"])


def main():
    st.set_page_config(page_title="ATS CV Evaluator", layout="wide")
    st.title("Evaluateur ATS de CV")
    st.write("Analyse un CV, mesure sa qualite et, si une offre est fournie, evalue sa compatibilite.")

    uploaded_resume = st.file_uploader("Importer un CV", ["pdf", "docx", "txt"])
    job_description = st.text_area("Coller la description du poste", height=220)

    if uploaded_resume is None:
        st.info("Importez un CV pour lancer l'analyse.")
        return

    try:
        resume_text = get_uploaded_file_text(uploaded_resume)
    except ValueError as error:
        st.error(str(error))
        return

    cleaned_resume = preprocess_text(resume_text)
    if not cleaned_resume:
        st.error("Impossible d'extraire du texte exploitable depuis ce fichier.")
        return

    with st.spinner("Analyse en cours..."):
        if job_description.strip():
            evaluation = evaluate_resume_against_job(cleaned_resume, job_description)
            render_results(evaluation, has_job_description=True)
        else:
            evaluation = evaluate_resume_only(cleaned_resume)
            render_results(evaluation, has_job_description=False)


if __name__ == "__main__":
    main()

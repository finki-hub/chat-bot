"""Residency-safe coarse topic labelling for chat queries (keyword match; label only)."""

OTHER_TOPIC = "other"

_TOPIC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "thesis_diploma",
        (
            "дипломск",
            "дипломир",
            "ментор",
            "магистер",
            "магистр",
            "докторск",
            "thesis",
            "diploma",
            "mentor",
        ),
    ),
    (
        "exams_grades",
        (
            "испит",
            "колоквиум",
            "оцен",
            "обврск",
            "положив",
            "паднав",
            "пријав",
            "рокови",
            "сесиј",
            "поправ",
            "exam",
            "grade",
            "colloqui",
            "midterm",
            "retake",
            "resit",
        ),
    ),
    (
        "schedule_timetable",
        (
            "распоред",
            "термин",
            "предавањ",
            "вежб",
            "часови",
            "сала",
            "schedule",
            "timetable",
            "lecture",
        ),
    ),
    (
        "admissions_enrollment",
        (
            "упис",
            "запишув",
            "конкурс",
            "бруцош",
            "прв циклус",
            "втор циклус",
            "enrol",
            "admiss",
            "applic",
            "freshman",
        ),
    ),
    (
        "fees_payments",
        (
            "уплат",
            "плаќањ",
            "школарин",
            "партиципациј",
            "стипенди",
            "цена",
            "чини",
            "кошта",
            "fee",
            "payment",
            "tuition",
            "scholarship",
            "refund",
        ),
    ),
    (
        "technical_account",
        (
            "лозинк",
            "најав",
            "профил",
            "налог",
            "платформ",
            "пристап",
            "courses.finki",
            "iknow",
            "moodle",
            "portal",
            "login",
            "log in",
            "password",
            "account",
        ),
    ),
    (
        "courses_syllabus",
        (
            "предмет",
            "силабус",
            "програм",
            "кредит",
            "ектс",
            "изборен",
            "course",
            "subject",
            "syllabus",
            "curricul",
            "ects",
            "elective",
        ),
    ),
    (
        "administration_documents",
        (
            "документ",
            "потврд",
            "уверение",
            "барањ",
            "индекс",
            "студентски прашањ",
            "архив",
            "печат",
            "молб",
            "document",
            "certificate",
            "transcript",
            "stamp",
        ),
    ),
    (
        "professors_contacts",
        (
            "професор",
            "асистент",
            "кабинет",
            "контакт",
            "мејл",
            "е-пошт",
            "professor",
            "assistant",
            "office hour",
            "contact",
            "email",
            "e-mail",
        ),
    ),
)


def classify_topic(query: str) -> str:
    text = query.casefold()
    for topic, keywords in _TOPIC_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return topic
    return OTHER_TOPIC


def classify_language(query: str) -> str:
    """Coarse script-based language label of ``query``: "mk", "en" or "other" (label only)."""
    cyrillic = sum("Ѐ" <= ch <= "ӿ" for ch in query)
    latin = sum(("a" <= ch <= "z") or ("A" <= ch <= "Z") for ch in query)
    if cyrillic == 0 and latin == 0:
        return "other"
    if cyrillic >= latin:
        return "mk"
    return "en"

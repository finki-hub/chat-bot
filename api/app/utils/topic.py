"""Lightweight, residency-safe topic classification for chat queries.

Maps a query to a coarse university-FAQ topic by keyword matching so analytics can group
questions by subject WITHOUT the raw query text ever leaving the server: only the resulting
``topic`` label is emitted. Covers Macedonian (Cyrillic) and English terms; extend
``_TOPIC_KEYWORDS`` to refine coverage. Returns ``OTHER_TOPIC`` when nothing matches.

Matching is plain casefolded substring containment, which catches Macedonian inflections
(``испит`` matches ``испитот`` / ``испити`` / ``испитна``) without a stemmer.
"""

OTHER_TOPIC = "other"

# Ordered: the first topic with a keyword hit wins, so the more specific intents (thesis,
# exams) are checked before the broader ones. ``technical_account`` precedes
# ``courses_syllabus`` so a login/password question that names ``courses.finki`` is not
# claimed by the ``course`` keyword. Keep keywords as stems so they match inflected forms
# via substring containment.
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
    """The coarse FAQ topic of ``query`` (a label only — the query text never leaves)."""
    text = query.casefold()
    for topic, keywords in _TOPIC_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return topic
    return OTHER_TOPIC

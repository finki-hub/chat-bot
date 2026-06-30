from enum import StrEnum


class QueryTransformMode(StrEnum):
    RAW = "raw"
    REWRITE = "rewrite"
    HYDE = "hyde"
    REWRITE_HYDE = "rewrite_hyde"

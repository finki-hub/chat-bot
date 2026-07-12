ALTER TABLE feedback
    ADD COLUMN dislike_reason_category TEXT,
    ADD COLUMN dislike_reason_detail TEXT,
    ADD CONSTRAINT feedback_dislike_reason_category_check
        CHECK (
            dislike_reason_category IS NULL
            OR dislike_reason_category IN ('incorrect', 'incomplete', 'off_topic', 'outdated', 'other')
        ),
    ADD CONSTRAINT feedback_dislike_reason_detail_length_check
        CHECK (
            dislike_reason_detail IS NULL
            OR char_length(dislike_reason_detail) <= 500
        );

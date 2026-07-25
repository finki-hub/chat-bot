UPDATE question
SET content = replace(
    content,
    '<@198249751001563136>',
    '[Delemangi](https://discord.com/users/198249751001563136)'
)
WHERE name = 'Што е ФИНКИ Хаб'
    AND strpos(content, '<@198249751001563136>') != 0;

UPDATE chat_message
SET content = replace(
        content,
        '<@198249751001563136>',
        '[Delemangi](https://discord.com/users/198249751001563136)'
    ),
    metadata = replace(
        metadata::TEXT,
        '<@198249751001563136>',
        '[Delemangi](https://discord.com/users/198249751001563136)'
    )::JSONB,
    parts = CASE
        WHEN parts IS NULL THEN NULL
        ELSE replace(
            parts::TEXT,
            '<@198249751001563136>',
            '[Delemangi](https://discord.com/users/198249751001563136)'
        )::JSONB
    END
WHERE role = 'assistant'
    AND (
        strpos(content, '<@198249751001563136>') != 0
        OR strpos(metadata::TEXT, '<@198249751001563136>') != 0
        OR strpos(COALESCE(parts::TEXT, ''), '<@198249751001563136>') != 0
    );

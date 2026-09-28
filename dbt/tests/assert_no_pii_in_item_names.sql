-- Guardrail behind the redaction step: published item names must not hold anything that
-- looks like a phone or card number (9+ digits in a row) or an email address.
select doc_id, line_idx, name
from {{ ref('fct_line_items') }}
where regexp_replace(name, '[\s.-]', '', 'g') ~ '\d{9,}'
   or name ~ '[[:alnum:]._%+-]+@[[:alnum:].-]+\.[[:alpha:]]{2,}'

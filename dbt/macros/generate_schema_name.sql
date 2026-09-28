-- Use the folder's schema as is (`staging`, `marts`) instead of dbt's default `public_staging`.
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ custom_schema_name if custom_schema_name else target.schema }}
{%- endmacro %}

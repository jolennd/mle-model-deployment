{# Build every model in one schema so tables are found by plain name: main unless DBT_SCHEMA is set. #}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ env_var('DBT_SCHEMA', 'main') }}
{%- endmacro %}

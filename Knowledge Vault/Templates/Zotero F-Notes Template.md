#### Date : {{importDate | format("DD-MM-YYYY") }}    Time : {{importDate | format("h:mm a") }}

Status : #concept

category Tag :

Tags :
# Fleeting Notes:
{%- macro calloutHeader(type, color) -%}  
{%- if type == "highlight" -%}  
<mark style="background-color: {{color}}">Note</mark>  
{%- endif -%}  
 
{%- if type == "text" -%}  
{%- endif -%}  
{%- endmacro -%}  
  
{% persist "annotations" %}  
{% set newAnnotations = annotations | filterby("date", "dateafter", lastImportDate) %}  
{% if newAnnotations.length > 0 %}  

 
  
{% for a in newAnnotations %}  
{{calloutHeader(a.type, a.color)}}  
 {{a.annotatedText}}  
{% endfor %}  
{% endif %}  
{% endpersist %}


---
## References:
[!Cite]
{{bibliography}}
-- Example model so `dbt build` has something to run. Delete it once you write real models.
select count(*) as n_complaints
from {{ source('raw', 'complaints') }}

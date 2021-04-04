-- задача 1.
with t as (
    select trn_id, acc_id, whs_id, trn_date, row_number() over (partition by acc_id order by trn_date) as rn
    from transactions
)
select acc_id, whs_id, trn_date
from t
where rn = 1;

select acc_id, whs_id, trn_date
from (select trn_id, acc_id, whs_id, trn_date, row_number() over (partition by acc_id order by trn_date) as rn
    from transactions) t
where rn = 1;

-- задача 2.
with t as (
    select trn_id, acc_id, whs_id, trn_date, row_number() over (partition by acc_id order by trn_date desc) as rn
    from transactions
)
select acc_id
from t
join warehouses wr on wr.whs_id = t.whs_id
where t.rn = 1
AND ((wr.frmt in (1,3) and t.trn_date < current_date - interval '8 week') OR (wr.frmt in (2) and t.trn_date < current_date - interval '4 week'))


select * from products pr
    join transactions tr on pr.trn_id = tr.trn_id and tr.acc_id = 1

-- задача 3.
with t as (
    select p.trn_id, p.art_id, qnty, row_number() over (partition by trn_id order by qnty) as rn
    from products p
    ),
tt as (
    select tr.acc_id, count(*) as cnt
    from transactions tr
    join t on t.trn_id = tr.trn_id and rn = 1
    where t.qnty >= 10
    group by tr.acc_id
),
total as (
    select acc_id, count(*) as cnt
    from transactions
    group by acc_id
)
select t.acc_id
from total t
join tt on tt.acc_id = t.acc_id
where cast(tt.cnt as decimal) / cast(t.cnt as decimal) >= 0.8

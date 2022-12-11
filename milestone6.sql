Select *
from
    (select first_name||' '||last_name as FullName,  to_char(sum(tip_amount), 'fmL99G999D00')as TotalServer_TipAmount
        from point_of_sales  ps join restaurant_servers rs
        on rs.server_id=ps.server_id
        group by rs.server_id,rs.first_name,rs.last_name
        order by TotalServer_TipAmount desc
    ) 
WHERE ROWNUM <= 3;

select distinct First_name||' '||Last_name as Host_FullName
from restaurant_hosts rh join (select  cwqi.customer_id,host_id
                                                        from customer_waiting_queue_info cwqi  join (select rpt.customer_id
                                                                                                                                        from receipts rpt join (select *
                                                                                                                                                  from(
                                                                                                                                                  select customer_id, hostc.custend_timein_wait - hostc.custbeg_timein_wait as difference
                                                                                                                                                  from hostscheck_customers hostc
                                                                                                                                                  order by difference)  
                                                                                                                                                 where rownum<=3)
                                                                                                                                                 custWait on rpt.customer_id=custWait.customer_id)
                                                                                            cusSer on cwqi.customer_id=cusSer.customer_id)
fastHost on rh.host_id=fastHost.host_id;

select *
from(
    select *
        from(
            select first_name||' '||last_name as Full_Name,total_years_of_experience
            from restaurant_hosts
            union
            select first_name||' '||last_name,total_years_of_experience
            from restaurant_kitchen_staff
            union
            select first_name||' '||last_name,total_yearsof_experience
            from restaurant_management
            union
            select first_name||' '||last_name,total_years_of_experience
            from restaurant_servers
            )
        order by total_years_of_experience desc
)
where rownum<=10;


select distinct First_name,Last_name,DOB,Total_years_of_experience
from 
(
    select server_id
    from 
    (
        select receipt_id
        from (select customer_id,alcohol_drink
                    from orders 
                    where alcohol_drink is not null) cusAl 
        join receipts r on cusAl.customer_id=r.customer_id) recAl
    join point_of_sales pos on recAl.receipt_id=pos.receipt_id) serAl
join restaurant_servers resSer on serAl.server_id=resSer.server_id
where extract(year from current_date)-extract(year from dob)<21;

Select ktin.kitchenstaff_id, first_name,last_name,TotalNum_IncorrectOrders
from 
(
    select kitchenstaff_id, count(kitchenstaff_id) as TotalNum_IncorrectOrders
    from orders
    where lower(order_accurate)='no'
    group by kitchenstaff_id
    having count(order_accurate)>=5
) ktin join restaurant_kitchen_staff rks on ktin.kitchenstaff_id=rks.kitchenstaff_id;    

select *
from
(
    select avg(hours_worked) as KitchenStaff_AverageHours
    from staff_schedules
    where kitchenstaff_id is not null
) ktch,
(
    select avg(hours_worked) as ManagementStaff_AverageHours
    from staff_schedules
    where management_id is not null
) mgs,
(
    select avg(hours_worked) as ServerStaff_AverageHours
    from staff_schedules
    where server_id is not null
) serv,
(
    select avg(hours_worked) as HostStaff_AverageHours
    from staff_schedules
    where host_id is not null
) host;

select main_course_meal_item,count(main_course_meal_item) as Freqeuncy_ItemOrdered
from orders
group by main_course_meal_item
having count(main_course_meal_item)>2
order by count(main_course_meal_item) desc;
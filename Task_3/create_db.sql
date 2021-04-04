create database lenta_db;

\c lenta_db;

create table warehouses (
    whs_id SERIAL PRIMARY KEY NOT NULL,
    frmt INTEGER NOT NULL,
    frmt_name TEXT
);

create table transactions (
    trn_id SERIAL PRIMARY KEY NOT NULL,
    acc_id INTEGER NOT NULL,
    whs_id INTEGER REFERENCES warehouses(whs_id),
    trn_date DATE,
    total FLOAT
);
create table products (
    trn_id INTEGER REFERENCES transactions(trn_id),
    art_id INTEGER NOT NULL,
    qnty FLOAT,
    value FLOAT
);

-- Insert some random data
insert into warehouses (frmt, frmt_name) values (1, 'У дома');
insert into warehouses (frmt, frmt_name) values (2, 'Авто');
insert into warehouses (frmt, frmt_name) values (3, 'Гипермаркет');
insert into transactions (acc_id, whs_id, trn_date, total) values (5, 2, '2021-03-14', 6372.6);
insert into transactions (acc_id, whs_id, trn_date, total) values (1, 3, '2021-03-15', 8621.9);
insert into transactions (acc_id, whs_id, trn_date, total) values (5, 2, '2021-01-12', 6492.4);
insert into transactions (acc_id, whs_id, trn_date, total) values (3, 3, '2021-03-02', 4839.5);
insert into transactions (acc_id, whs_id, trn_date, total) values (3, 2, '2021-01-28', 5183.3);
insert into transactions (acc_id, whs_id, trn_date, total) values (1, 3, '2021-02-28', 6664.2);
insert into transactions (acc_id, whs_id, trn_date, total) values (4, 3, '2021-03-30', 9411.6);
insert into transactions (acc_id, whs_id, trn_date, total) values (3, 1, '2021-01-06', 3454.5);
insert into transactions (acc_id, whs_id, trn_date, total) values (5, 3, '2021-01-11', 7054.7);
insert into transactions (acc_id, whs_id, trn_date, total) values (1, 3, '2021-01-13', 6282.4);
insert into transactions (acc_id, whs_id, trn_date, total) values (2, 3, '2021-01-03', 3349.5);
insert into transactions (acc_id, whs_id, trn_date, total) values (1, 1, '2021-02-17', 5943.8);
insert into transactions (acc_id, whs_id, trn_date, total) values (1, 2, '2021-02-03', 9925.2);
insert into transactions (acc_id, whs_id, trn_date, total) values (5, 1, '2021-02-10', 6900.4);
insert into transactions (acc_id, whs_id, trn_date, total) values (5, 1, '2021-03-28', 2256.7);
insert into transactions (acc_id, whs_id, trn_date, total) values (1, 1, '2021-03-20', 5553.7);
insert into transactions (acc_id, whs_id, trn_date, total) values (1, 3, '2021-02-18', 637.7);
insert into transactions (acc_id, whs_id, trn_date, total) values (2, 1, '2021-03-12', 78.1);
insert into transactions (acc_id, whs_id, trn_date, total) values (2, 3, '2021-03-19', 1144.5);
insert into transactions (acc_id, whs_id, trn_date, total) values (2, 2, '2021-03-16', 6423.8);
insert into transactions (acc_id, whs_id, trn_date, total) values (4, 3, '2021-03-07', 6952.6);
insert into transactions (acc_id, whs_id, trn_date, total) values (1, 3, '2021-03-21', 1332.8);
insert into transactions (acc_id, whs_id, trn_date, total) values (5, 2, '2021-01-13', 8687.6);
insert into transactions (acc_id, whs_id, trn_date, total) values (1, 1, '2021-01-28', 6908.0);
insert into transactions (acc_id, whs_id, trn_date, total) values (5, 3, '2021-03-15', 251.4);
insert into transactions (acc_id, whs_id, trn_date, total) values (2, 2, '2021-03-08', 7667.2);
insert into transactions (acc_id, whs_id, trn_date, total) values (5, 2, '2021-03-17', 887.8);
insert into transactions (acc_id, whs_id, trn_date, total) values (2, 1, '2021-02-10', 4302.4);
insert into transactions (acc_id, whs_id, trn_date, total) values (2, 2, '2021-03-10', 1182.8);
insert into transactions (acc_id, whs_id, trn_date, total) values (4, 1, '2021-02-20', 9793.3);
insert into transactions (acc_id, whs_id, trn_date, total) values (4, 1, '2021-01-13', 4038.1);
insert into transactions (acc_id, whs_id, trn_date, total) values (3, 1, '2021-02-28', 4855.5);
insert into transactions (acc_id, whs_id, trn_date, total) values (3, 2, '2021-01-17', 9278.6);
insert into transactions (acc_id, whs_id, trn_date, total) values (1, 3, '2021-02-13', 3394.7);
insert into transactions (acc_id, whs_id, trn_date, total) values (1, 2, '2021-02-03', 9915.4);
insert into transactions (acc_id, whs_id, trn_date, total) values (1, 2, '2021-03-15', 5832.1);
insert into transactions (acc_id, whs_id, trn_date, total) values (5, 2, '2021-02-24', 6631.5);
insert into transactions (acc_id, whs_id, trn_date, total) values (3, 3, '2021-03-13', 7199.3);
insert into transactions (acc_id, whs_id, trn_date, total) values (5, 2, '2021-02-11', 3526.2);
insert into transactions (acc_id, whs_id, trn_date, total) values (5, 2, '2021-02-23', 9403.8);
insert into transactions (acc_id, whs_id, trn_date, total) values (2, 3, '2021-02-06', 8948.9);
insert into transactions (acc_id, whs_id, trn_date, total) values (1, 2, '2021-04-01', 9685.8);
insert into transactions (acc_id, whs_id, trn_date, total) values (2, 2, '2021-02-19', 8943.0);
insert into transactions (acc_id, whs_id, trn_date, total) values (4, 2, '2021-02-19', 4389.2);
insert into transactions (acc_id, whs_id, trn_date, total) values (3, 3, '2021-01-29', 304.0);
insert into transactions (acc_id, whs_id, trn_date, total) values (4, 1, '2021-03-08', 169.6);
insert into transactions (acc_id, whs_id, trn_date, total) values (2, 1, '2021-01-10', 5538.5);
insert into transactions (acc_id, whs_id, trn_date, total) values (3, 1, '2021-01-18', 1035.0);
insert into transactions (acc_id, whs_id, trn_date, total) values (3, 2, '2021-02-09', 7073.5);
insert into transactions (acc_id, whs_id, trn_date, total) values (3, 1, '2021-01-13', 53.0);
insert into products (trn_id, art_id, qnty, value) values (26, 22, 13, 25.8);
insert into products (trn_id, art_id, qnty, value) values (34, 6, 12, 539.1);
insert into products (trn_id, art_id, qnty, value) values (5, 13, 2, 293.0);
insert into products (trn_id, art_id, qnty, value) values (12, 13, 15, 507.2);
insert into products (trn_id, art_id, qnty, value) values (23, 8, 20, 267.6);
insert into products (trn_id, art_id, qnty, value) values (3, 16, 4, 327.6);
insert into products (trn_id, art_id, qnty, value) values (20, 2, 4, 285.8);
insert into products (trn_id, art_id, qnty, value) values (6, 18, 12, 24.0);
insert into products (trn_id, art_id, qnty, value) values (43, 21, 7, 535.4);
insert into products (trn_id, art_id, qnty, value) values (45, 16, 12, 522.2);
insert into products (trn_id, art_id, qnty, value) values (39, 19, 16, 374.7);
insert into products (trn_id, art_id, qnty, value) values (39, 10, 12, 458.1);
insert into products (trn_id, art_id, qnty, value) values (12, 25, 15, 468.0);
insert into products (trn_id, art_id, qnty, value) values (18, 3, 2, 434.2);
insert into products (trn_id, art_id, qnty, value) values (14, 9, 1, 115.4);
insert into products (trn_id, art_id, qnty, value) values (45, 5, 1, 288.0);
insert into products (trn_id, art_id, qnty, value) values (8, 17, 5, 262.0);
insert into products (trn_id, art_id, qnty, value) values (39, 13, 6, 371.4);
insert into products (trn_id, art_id, qnty, value) values (32, 5, 1, 545.8);
insert into products (trn_id, art_id, qnty, value) values (12, 7, 8, 58.8);
insert into products (trn_id, art_id, qnty, value) values (43, 14, 3, 387.3);
insert into products (trn_id, art_id, qnty, value) values (27, 16, 8, 284.8);
insert into products (trn_id, art_id, qnty, value) values (8, 23, 11, 366.9);
insert into products (trn_id, art_id, qnty, value) values (25, 16, 12, 407.6);
insert into products (trn_id, art_id, qnty, value) values (27, 21, 1, 226.9);
insert into products (trn_id, art_id, qnty, value) values (39, 19, 10, 319.6);
insert into products (trn_id, art_id, qnty, value) values (20, 19, 15, 66.7);
insert into products (trn_id, art_id, qnty, value) values (43, 21, 4, 145.5);
insert into products (trn_id, art_id, qnty, value) values (12, 1, 19, 383.4);
insert into products (trn_id, art_id, qnty, value) values (19, 1, 4, 136.6);
insert into products (trn_id, art_id, qnty, value) values (34, 5, 10, 13.0);
insert into products (trn_id, art_id, qnty, value) values (35, 15, 17, 96.8);
insert into products (trn_id, art_id, qnty, value) values (18, 1, 1, 93.5);
insert into products (trn_id, art_id, qnty, value) values (34, 8, 20, 428.2);
insert into products (trn_id, art_id, qnty, value) values (17, 8, 7, 348.5);
insert into products (trn_id, art_id, qnty, value) values (17, 27, 7, 130.1);
insert into products (trn_id, art_id, qnty, value) values (7, 30, 2, 272.4);
insert into products (trn_id, art_id, qnty, value) values (3, 22, 8, 245.9);
insert into products (trn_id, art_id, qnty, value) values (40, 2, 17, 300.4);
insert into products (trn_id, art_id, qnty, value) values (43, 10, 2, 326.7);
insert into products (trn_id, art_id, qnty, value) values (32, 23, 5, 483.0);
insert into products (trn_id, art_id, qnty, value) values (24, 16, 16, 361.9);
insert into products (trn_id, art_id, qnty, value) values (15, 4, 3, 143.8);
insert into products (trn_id, art_id, qnty, value) values (25, 11, 9, 130.0);
insert into products (trn_id, art_id, qnty, value) values (17, 12, 16, 23.3);
insert into products (trn_id, art_id, qnty, value) values (23, 3, 10, 69.8);
insert into products (trn_id, art_id, qnty, value) values (41, 5, 3, 38.4);
insert into products (trn_id, art_id, qnty, value) values (26, 8, 9, 246.6);
insert into products (trn_id, art_id, qnty, value) values (13, 20, 1, 230.4);
insert into products (trn_id, art_id, qnty, value) values (47, 4, 10, 62.4);

import psycopg2 as psql
from Postgres_Controller import PostgresConnection

#region Create table commands

percentile_groups_table = '''
    DROP TABLE IF EXISTS percentil_groups CASCADE;
    
    CREATE TABLE percentil_groups (
        id SERIAL PRIMARY KEY,
        project_id VARCHAR(255) NOT NULL,
        metadata JSON,
        number_genes INTEGER NOT NULL,
        number_cells INTEGER NOT NULL
    );
'''

percentiles_table = '''
    DROP TABLE IF EXISTS percentiles CASCADE;
    
    CREATE TABLE percentiles (
        id SERIAL PRIMARY KEY,
        gene_name VARCHAR(255) NOT NULL,
        percentile float NOT NULL,
        percentil_group INTEGER NOT NULL,
        FOREIGN KEY (percentil_group) 
            REFERENCES percentil_groups (id) 
            ON UPDATE CASCADE ON DELETE CASCADE
    );
'''

gcn_table = '''
    DROP TABLE IF EXISTS gcn CASCADE;

    CREATE TABLE gcn (
        id SERIAL PRIMARY KEY,
        project_id VARCHAR(255) NOT NULL,
        correction VARCHAR(255) NOT NULL,
        iter_pseudocells INTEGER NOT NULL,
        metadata JSON
    );
'''

modules_table = '''
    DROP TABLE IF EXISTS modules CASCADE;

    CREATE TABLE modules (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        gcn INTEGER NOT NULL,
        FOREIGN KEY (gcn) 
            REFERENCES gcn (id) 
            ON UPDATE CASCADE ON DELETE CASCADE
    );
'''

module_membership_table = '''
    DROP TABLE IF EXISTS module_membership CASCADE;

    CREATE TABLE module_membership (
        id SERIAL PRIMARY KEY,
        module INTEGER NOT NULL,
        gene_name VARCHAR(255) NOT NULL,
        MM float NOT NULL,
        FOREIGN KEY (module) 
            REFERENCES modules (id) 
            ON UPDATE CASCADE ON DELETE CASCADE
    );
'''

term_table = '''
    DROP TABLE IF EXISTS term CASCADE;

    CREATE TABLE term (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        term_id VARCHAR(255) NOT NULL,
        source VARCHAR(255) NOT NULL,
        IC float NULL
    );
'''

annotation_table = '''
    DROP TABLE IF EXISTS annotation CASCADE;

    CREATE TABLE annotation (
        id SERIAL PRIMARY KEY,
        module INTEGER NOT nULL,
        term INTEGER NOT NULL,
        p_value float NOT NULL,
        FOREIGN KEY (module) 
            REFERENCES modules (id) 
            ON UPDATE CASCADE ON DELETE CASCADE,
        FOREIGN KEY (term) 
            REFERENCES term (id) 
            ON UPDATE CASCADE ON DELETE CASCADE
    );
'''


#endregion

commands = [
    percentile_groups_table,
    percentiles_table,
    gcn_table,
    modules_table,
    module_membership_table,
    term_table,
    annotation_table
]

# Running all commands in postgres
with PostgresConnection() as conn:
    cur = conn.cursor()

    # create table one by one
    for command in commands:
        cur.execute(command)
    # commit the changes
    conn.commit()

    # read tables
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)

    tables = cur.fetchall()

    if len(tables) == len(commands):
        print('Tables have been correctly created')
    else:
        print('Some error has occur during tables creation')

    # close communication with the PostgreSQL database server
    cur.close()

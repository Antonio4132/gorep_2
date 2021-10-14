import psycopg2
from io import StringIO


class PostgresConnection(object):

    def __enter__(self):
        # connect to the PostgreSQL server
        self.conn = psycopg2.connect(
            host="localhost",
            database="sc-db",
            user="sc-user",
            password="single-cell21."
        )

        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
        

class PostgresController(object):

    def __init__(self):
        super()

    def test_con(self):
        with PostgresConnection() as conn:
            cur = conn.cursor()

            print('PostgreSQL database version:')

            cur.execute('SELECT version()')
            db_version = cur.fetchone()
            print(db_version)

            cur.close()

    def get_percentile_group_id(self, project_id, metadata, number_genes, number_cells):
        metadata_str = str(metadata).replace("'", '"')

        command = f"""
            SELECT id
            FROM percentil_groups
            WHERE
                project_id = '{project_id}' AND
                metadata::jsonb @> '{metadata_str}'::jsonb AND
                '{metadata_str}'::jsonb @>  metadata::jsonb AND
                number_genes = {number_genes} AND
                number_cells = {number_cells}
        """

        with PostgresConnection() as conn:
            cur = conn.cursor()
            # read tables
            cur.execute(command)
            percentile_group_id = cur.fetchone()
            # close communication with the PostgreSQL database server
            cur.close()
            # commit the changes
            conn.commit()

        if percentile_group_id is not None:
            return percentile_group_id[0]
            

    def add_percentile_group(self, project_id, metadata, number_genes, number_cells):
        percentile_group_id = self.get_percentile_group_id(project_id, metadata, number_genes, number_cells)
        metadata_str = str(metadata).replace("'", '"')
        
        if percentile_group_id is not None:
            return percentile_group_id

        command = f"""
            INSERT INTO percentil_groups (project_id, 
                                     metadata,
                                     number_genes,
                                     number_cells)
            VALUES ('{project_id}', 
                    '{metadata_str}',
                    {number_genes},
                    {number_cells})

            RETURNING id;
        """

        with PostgresConnection() as conn:
            cur = conn.cursor()
            
            # execute command
            cur.execute(command)
            percentile_group_id = cur.fetchone()

            # close communication with the PostgreSQL database server
            cur.close()
            
            # commit the changes
            conn.commit()
        
        return percentile_group_id[0]
    
    
    def add_percentile_with_group(self, project_id, gene_name, percentile, number_genes, number_cells, metadata={}):
        percentile_id = -1

        percentile_group_id = self.add_percentile_group(project_id, metadata, number_genes, number_cells)[0]

        command = f"""
            INSERT INTO percentiles (gene_name,
                                     percentile,
                                     percentil_group)
            VALUES ('{gene_name}', 
                    {percentile},
                    {percentile_group_id})

            RETURNING id;
        """

        with PostgresConnection() as conn:
            cur = conn.cursor()

            # read tables
            cur.execute(command)
            percentile_id = cur.fetchone()
            # close communication with the PostgreSQL database server
            cur.close()
            # commit the changes
            conn.commit()

        return percentile_id[0], percentile_group_id
    
    def add_percentile(self, gene_name, percentile, percentile_group_id):
        command = f"""
            INSERT INTO percentiles (gene_name,
                                     percentile,
                                     percentil_group)
            VALUES ('{gene_name}', 
                    {percentile},
                    {percentile_group_id})

            RETURNING id;
        """

        with PostgresConnection() as conn:
            cur = conn.cursor()

            # read tables
            cur.execute(command)
            percentile_id = cur.fetchone()
            # close communication with the PostgreSQL database server
            cur.close()
            # commit the changes
            conn.commit()

        return percentile_id[0]
    
    
    def add_sampling_percentiles(self, percentiles, sampling_info, method):
        project_id = sampling_info['project_id'].iloc[0]
        metadata = sampling_info['metadata'].iloc[0]
        number_genes = sampling_info['number_genes'].iloc[0]
        number_cells = sampling_info['number_cells'].iloc[0]
        
        percentile_group_id = self.add_percentile_group(project_id, metadata, number_genes, number_cells)

        method(percentiles, percentile_group_id)

        return percentile_group_id
    
    
    def simple_insert_percentile(self, percentiles, percentile_group_id):
        for _, row in percentiles.iterrows():
            gene_name = row['gene_name']
            percentile = row['percentile']

            self.add_percentile(gene_name, 
                                percentile, 
                                percentile_group_id)
            
     
    def insert_many_percentile(self, percentiles, percentile_group_id):
        """
        Using cursor.executemany() to insert the dataframe
        """
        # Create a list of tupples from the dataframe values
        tuples = [tuple(x) for x in percentiles.to_numpy()]
        
        # Comma-separated dataframe columns
        cols = ','.join(list(percentiles.columns)) + ',percentil_group'
        
        # SQL quert to execute
        query  = f"INSERT INTO percentiles ({cols}) VALUES(%s,%f,{percentile_group_id})"
        
        with PostgresConnection() as conn:
            
            cursor = conn.cursor()
            try:
                cursor.executemany(query, tuples)
                conn.commit()

            except (Exception, psycopg2.DatabaseError) as error:
                print("Error: %s" % error)
                conn.rollback()
                cursor.close()
                return 1

            cursor.close()
            
    def copy_from_stringio_percentile(self, percentiles, percentile_group_id):
        """
        Here we are going save the dataframe in memory 
        and use copy_from() to copy it to the table
        """
        percentiles['percentil_group'] = [percentile_group_id] * len(percentiles)
        # save dataframe to an in memory buffer
        buffer = StringIO()
        percentiles.to_csv(buffer, header=False, index=False)
        buffer.seek(0)

        with PostgresConnection() as conn:

            cursor = conn.cursor()
            try:
                cursor.copy_from(buffer, 'percentiles', sep=",", columns=list(percentiles.columns))
                conn.commit()
            except (Exception, psycopg2.DatabaseError) as error:
                print("Error: %s" % error)
                conn.rollback()
                cursor.close()
                return 1
            cursor.close()
    

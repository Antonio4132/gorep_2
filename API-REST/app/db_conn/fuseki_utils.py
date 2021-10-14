import pandas as pd
URI_prefix = 'http://www.semanticweb.org/alicia/ontologies/2020/8/singleCellRepositories#'


def fuseki_to_dict(response, dict_type='list'):
    headers = response.json()["head"]["vars"]
    results = response.json()["results"]['bindings']

    if not results:
        return []

    rows = []

    for result in results:
        result_dict = {}
        for header in headers:
            if header not in result:
                result_dict[header] = None
            else:
                value = result[header]['value']

                if URI_prefix in value:
                    value = value.split('#')[-1]
                result_dict[header] = value

        rows.append(result_dict)

    df = pd.DataFrame(rows)

    if 'project_ID' not in df.columns:
        df_dict = df.to_dict(dict_type)

        if dict_type == 'list':
            for key in df_dict:
                l = list(set(df_dict[key]))
                df_dict[key] = l[0] if len(l) == 1 else l
        elif dict_type == 'records':
            aux = {}

            for item in df_dict:

                term = item['term']
                URI = item['URI']

                if term not in aux:
                    aux[term] = []

                aux[term].append(URI)

            df_dict = []

            for term, URIs in aux.items():
                df_dict.append({
                    'term': term,
                    'URI': URIs
                })

        return df_dict

    projects = []
    for project_ID, data in df.groupby('project_ID'):
        project = {
            'project_ID': project_ID,
        }

        for column in [x for x in data if x != 'project_ID']:
            column_data = data[column].unique().tolist()
            if len(column_data) == 1:
                column_data = column_data[0]

            if column_data is not None:
                project[column] = column_data

        projects.append(project)

    return projects


def fuseki_to_list(response):
    items = []
    header = response.json()["head"]["vars"][0]
    results = response.json()["results"]['bindings']
    for result in results:
        item = result[header]['value']
        if URI_prefix in item:
            item = item.split('#')[-1]
        items.append(item)

    return items
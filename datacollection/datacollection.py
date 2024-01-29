import requests
import pandas as pd


def clean_data(x):
    return str(x).replace('\r', '').replace('\n', '')


def get_all_tensorflow_bug_reports(api_token):
    base_url = 'https://api.github.com/repos/tensorflow/tensorflow/issues'
    headers = {'Authorization': f'token {api_token}'}

    all_bug_reports = []

    for page_number in range(1, 77):
        url = f'{base_url}?page={page_number}&q=is%3Aissue+is%3Aopen+label%3Atype%3Abug'
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            bug_reports_on_page = response.json()
            all_bug_reports.extend(bug_reports_on_page)
        else:
            print(f"Error: Unable to fetch bug reports on page {page_number} (Status Code: {response.status_code})")

    bug_reports_df = pd.DataFrame(all_bug_reports)
    bug_reports_df = bug_reports_df.applymap(clean_data)
    bug_reports_df.to_excel('C:/Users/Liyujie/Desktop/tensorflow_bug_reports.xlsx', index=False)

api_token = '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'
get_all_tensorflow_bug_reports(api_token)

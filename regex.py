import json
import re
org_string = """
Selamat ya Mas @adietaufan !! #selamat... https://www.instagram.com/p/btddzj3jo9a
"""
pattern = r'(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)'

mod_string = re.sub(pattern, '', org_string)
print(mod_string)

# pattern = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)\w+'
# pattern = r'@\w+'
# mod_string = re.sub(pattern, '', org_string)

# pattern = r'#\w+'
# mod_string = re.sub(pattern, '', mod_string)

# print(mod_string)


# def remove_regex(json_file):
#     with open(json_file) as f:
#         data = json.load(f)
#     regex = [r'#\w+', r'@\w+']
#     for i in data:
#         regex.append(i['regex'])
#     return regex


# file_json = 'data.json'
# print(remove_regex(file_json))

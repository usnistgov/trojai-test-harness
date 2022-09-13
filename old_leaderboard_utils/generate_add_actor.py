from leaderboards import json_io


def generate_add_actor(args):
    actor_json = json_io.read(args.actor_json_filepath)

    for email, actor in actor_json['_ActorManager__actors'].items():
        actor_email = actor['email']
        actor_name = actor['name']
        actor_poc_email = actor['poc_email']
        actor_type = 'contractor'

        print('python actor.py add-actor --trojai-config-filepath {} --name {} --email {} --poc-email {} --type {}'.format(args.trojai_config_filepath, actor_name, actor_email, actor_poc_email, actor_type))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Loads a prior round actor.json and formulates the command-line for adding them to the new round')

    parser.add_argument('--trojai-config-filepath', type=str, help='The filepath to the trojai config', required=True)
    parser.add_argument('--actor-json-filepath', type=str, help='The filepath to the actor json from the prior round', required=True)
    args = parser.parse_args()
    generate_add_actor(args)
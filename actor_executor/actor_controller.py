# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import logging

from actor_executor.actor import ActorManager
from actor_executor.config import Config


def remove_actor(key, config):
    actors = ActorManager.load_json(config.actor_json_file)

    logging.info("Removing actor: " + str(key))
    if key not in actors.get_keys():
        logging.error('Unable to remove actor: {}, they are not currently in the list of actors: {}'.format(key, actors.get_keys()))
        raise IOError('Unable to remove actor: {}, they are not currently in the list of actors: {}'.format(key, actors.get_keys()))

    actors.remove_actor(key)

    # write the update actors back to disk
    actors.save_json(config.actor_json_file)

    logging.info("Successfully removed actor " + str(key))


def reset_actor(key, config):
    actors = ActorManager.load_json(config.actor_json_file)

    logging.info("Resetting actor: " + str(reset_str))
    if key not in actors.get_keys():
        logging.error('Unable to reset actor: {}, they are not currently in the list of actors: {}'.format(key, actors.get_keys()))
        raise IOError('Unable to reset actor: {}, they are not currently in the list of actors: {}'.format(key, actors.get_keys()))

    actor = actors.get(key)
    actor.reset()

    # write the update actors back to disk
    actors.save_json(config.actor_json_file)

    logging.info("Successfully reset actor " + key)


def add_actor(add_str, config):
    actors = ActorManager.load_json(config.actor_json_file)

    logging.info("Adding actor: " + str(add_str))
    items = add_str.split(',')
    team_name = items[0]
    team_email = items[1]
    poc_email = items[2]
    # add actor to Manger
    actors.add_actor(team_email, team_name, poc_email) # raises RuntimeException if actor already exists

    # write the update actors back to disk
    actors.save_json(config.actor_json_file)

    logging.info("Successfully added actor {}".format(add_str))


def convert_to_csv(output_file, config):
    actors = ActorManager.load_json(config.actor_json_file)
    write_header = True

    with open(output_file, 'w') as file:
        for actorKey in actors.get_keys():
            actor = actors.get(actorKey)
            if write_header:
                for key in actor.__dict__.keys():
                    file.write(key + ',')
                file.write('\n')
                write_header = False

            for key in actor.__dict__.keys():
                file.write(str(actor.__dict__[key]) + ',')
            file.write('\n')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Manage the addition and removal of TrojAI Actors (challenge participant teams)')

    parser.add_argument('--add-actor', type=str,
                        help='Adds an actor in CSV "ActorTeamName,ActorTeamEmail,PocEmail", this will exit after adding the actor',
                        default=None)
    parser.add_argument("--remove-actor", type=str,
                        help='Removes an actor (based on actor team name), this will exit after removing the actor (the repo that is owned by the actor will not be deleted)',
                        default=None)
    parser.add_argument("--reset-actor", type=str,
                         help="Resets a team to allow them to resubmit (based on actor team name)",
                         default=None)
    parser.add_argument('--config-file', type=str,
                        help='The JSON file that describes all actors ',
                        default='config.json')
    parser.add_argument('--log-file', type=str,
                        help='The log file',
                        default="actor-manager.log")

    parser.add_argument('--convert-to-csv', type=str,
                        help='The file to save actors to CSV (converts json to CSV)',
                        default=None)

    parser.set_defaults(push_html=True)

    args = parser.parse_args()

    config = Config.load_json(args.config_file)

    add_str = args.add_actor
    remove_str = args.remove_actor
    reset_str = args.reset_actor
    log_file = args.log_file
    convert_to_csv_output_file = args.convert_to_csv

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    if add_str is not None:
        add_actor(add_str, config)
    if remove_str is not None:
        remove_actor(remove_str, config)
    if reset_str is not None:
        reset_actor(reset_str, config)
    if convert_to_csv_output_file is not None:
        convert_to_csv(convert_to_csv_output_file, config)

import os
import configparser

# Configuration file name
BASE_CONFIG = "config.ini"
TEMPLATE_CONFIG = "template_config.ini"

# Default configuration values
DEFAULT_VALUES = {
    "llamafactory": {
        "cli_port": "8000",
    }
}
'''
TODO: 未实现template_config.ini的初始化等操作，只实现查询
'''

def ensure_config_file():
    """Ensure the configuration file exists. If not, create it with default values."""
    if not os.path.exists(BASE_CONFIG):
        create_default_config()

def create_default_config():
    """Create a configuration file with default values."""
    config = configparser.ConfigParser()
    for section, values in DEFAULT_VALUES.items():
        config[section] = values
    save_config(config)

def load_config(config_type):
    """Load the configuration file and return the ConfigParser object."""
    config = configparser.ConfigParser()
    if config_type == "base_config":
        config.read(BASE_CONFIG)
    elif config_type == "template_config":
        config.read('template_config.ini')

    return config

def get(config_type, section, key):
    """Get the value of a key in a section. Raise KeyError if not found."""
    config = load_config(config_type)
    if section not in config or key not in config[section]:
        raise KeyError(f"No such key: [{section}] {key}")
    return config[section][key]

def set(section, key, value):
    """Set the value of a key in a section. Create the section if it does not exist."""
    config = load_config()
    if section not in config:
        config.add_section(section)
    config[section][key] = value
    save_config(config)

def save_config(config):
    """Save the current configuration to the file."""
    with open(BASE_CONFIG, "w") as configfile:
        config.write(configfile)

# Ensure the config file exists on import
ensure_config_file()

# Usage example
if __name__ == "__main__":
    # Read a value (will raise KeyError if the key does not exist)
    try:
        print(get("DEFAULT", "setting1"))
    except KeyError as e:
        print(e)

    # Set a new value
    set("NEW_SECTION", "new_key", "new_value")

    # Read the newly set value
    print(get("NEW_SECTION", "new_key"))

"""Other methods common to both the main package and some sample analysis script(s)."""


def get_temperatures(minimum_temperature, maximum_temperature, number_of_temperature_values):
    """Creates a list of the temperatures over which super-aLby iterates"""
    if minimum_temperature < 0.0:
        raise ValueError("Give a value not less than 0.0 as minimum_temperature in other_methods.get_temperatures().")
    if maximum_temperature < 0.0:
        raise ValueError("Give a value not less than 0.0 as maximum_temperature in other_methods.get_temperatures().")
    if maximum_temperature < minimum_temperature:
        raise ValueError("Give values of minimum_temperature and maximum_temperature in "
                         "other_methods.get_temperatures() such that the value of maximum_temperature is not less than "
                         "the value of minimum_temperature..")
    if number_of_temperature_values < 1:
        raise ValueError("Give a value not less than 1 as number_of_temperature_values in "
                         "other_methods.get_temperatures().")
    if number_of_temperature_values == 1 and minimum_temperature != maximum_temperature:
        raise ValueError("As the value of number_of_temperature_values is equal to 1, give equal values of "
                         "minimum_temperature and maximum_temperature in other_methods.get_temperatures().")
    if number_of_temperature_values == 1:
        return [minimum_temperature]
    temperature_increment = (maximum_temperature - minimum_temperature) / (number_of_temperature_values - 1)
    return [minimum_temperature + temperature_increment * temperature_index
            for temperature_index in range(number_of_temperature_values)]

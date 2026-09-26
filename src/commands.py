# src/commands.py
COMMANDS = {
    "fire": "Analyze the area for fire. Locate the source and report the extent.",
    "buildings": "Identify and count all buildings in the area. Provide coordinates.",
    "robbers": "Search for a white car that is moving away from the area. Report its position.",
    "oil_spill": "Detect any signs of an oil spill in the water bodies or on the ground.",
    "water_source": "We have to find some huge water source for firefighters.",
    "car_accident": "A car accident has occurred. The police are on the scene.",
    "missing_person": "A missing person has been reported. The police are searching for them." ,
    "hogweed_thickets": "Hogweed thickets are growing in the area.",
    "plant_diseases": "Plant diseases are spreading in the area.",
    "unauthorized_festival": "Unauthorized festival is taking place in the area.",
    "rally": "A rally is taking place in the area.",
    "illegal_pipeline_tap": "Illegal pipeline tap has been detected in the area.",
    "plane_crash": "A plane crash has occurred. The rescue team is on the scene.",
    "military_invasion": "Military invasion has been detected in the area.",
    "cults": "Cults are gathering in the area.",
    "lost_cow": "Lost cow has been reported in the area.",
    "illegal_migrants": "Illegal migrants have been detected in the area.",
    "meteorite": "Meteorite has been detected in the area.",
    "spacecraft_landed": "Spacecraft landed in the area.",
    "illegal_constructions": "Illegal constructions have been detected in the area.",
    "buildings_fire": "Building fire has been detected in the area."
}
# water_source car_accident hogweed_thickets oil_spill unauthorized_festival
def get_command_description(command_key: str) -> str:
    """Возвращает описание для агента по короткому ключу."""
    return COMMANDS.get(command_key, "Analyze the area for any unusual activity.")
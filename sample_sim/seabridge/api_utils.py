import requests


class APIUtils:
    api_url = 'http://127.0.0.1:5000/api/'

    @classmethod
    def _get_extension_for_object_type(cls, object_type):
        if object_type == 'Waypoint':
            return 'waypoint'
        elif object_type == 'RobotState':
            return 'robotstate/'
        elif object_type == 'Image':
            return 'image/'
        elif object_type == 'SquareWorkspace':
            return "workspace"
        else:
            raise ValueError(f"Object type not understood: {object_type}")

    @classmethod
    def _get_object_json(cls, object_type, object_id, api_url):
        extension = cls._get_extension_for_object_type(object_type)
        url = api_url + extension + str(object_id)
        response = requests.get(url=url)
        return response.json()

    @classmethod
    def _post_object(cls, object_type, api_url, json=None, data=None, files=None):
        extension = cls._get_extension_for_object_type(object_type)
        url = api_url + extension
        if json is not None:
            if data is not None or files is not None:
                raise ValueError("data and/or files is specified but json is already specified; should only specify either (json) or (data and files)")
            response = requests.post(url=url, json=json)
        elif data is not None and files is not None:
            if json is not None:
                raise ValueError("All of data and files and json are specified; should only specify either (json) or (data and files)")
            response = requests.post(url=url, files=files, data=data) 
        elif files is not None:
            response = requests.post(url=url, files=files)
        else:
            raise ValueError("Either (json) or (data and files) should be specified")
        return response.json()



    @classmethod
    def _get_all_objects_json(cls, object_type, api_url):
        extension = cls._get_extension_for_object_type(object_type)
        url = api_url + extension
        response = requests.get(url=url)
        return response.json()

    # Get One methods
    @classmethod
    def get_waypoint_json(cls, id, api_url=api_url):
        return cls._get_object_json('Waypoint', id, api_url)

    @classmethod
    def get_robotstate_json(cls, id, api_url=api_url):
        return cls._get_object_json('RobotState', id, api_url)

    @classmethod
    def get_image_json(cls, id, api_url=api_url):
        return cls._get_object_json('Image', id, api_url)

    # Post methods
    @classmethod
    def post_waypoint_json(cls, json, api_url=api_url):
        return cls._post_object('Waypoint', api_url, json=json)

    @classmethod
    def post_robotstate_json(cls, json, api_url=api_url):
        return cls._post_object('RobotState', api_url, json=json)

    @classmethod
    def post_image_json(cls, data, files, api_url=api_url):
        return cls._post_object('Image', api_url, data=data, files=files)

    # Get All methods
    @classmethod
    def get_all_workspaces_json(cls, api_url=api_url):
        return cls._get_all_objects_json('SquareWorkspace', api_url)

    @classmethod
    def get_all_waypoints_json(cls, api_url=api_url):
        return cls._get_all_objects_json('Waypoint', api_url)

    @classmethod
    def get_all_robotstates_json(cls, api_url=api_url):
        return cls._get_all_objects_json('RobotState', api_url)

    @classmethod
    def get_all_images_json(cls, api_url=api_url):
        return cls._get_all_objects_json('Image', api_url)

    @classmethod
    def request_succeeded(cls, json):
        return 'error' not in json

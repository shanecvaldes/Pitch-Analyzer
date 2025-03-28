from flask import Flask, request, jsonify, session, render_template, redirect, request, send_file, send_from_directory
from flask_cors import CORS
import os
import glob
app = Flask(__name__)
CORS(app)

@app.route('/get/graphic_paths/search', methods=['GET'])
def get_graphic_paths():
    type_query = request.args.get('type', '').strip().replace(' ', '_')
    # print(type_query)
    pitchTypes = set()
    standardizes = set()
    groupings = set()
    dataCategories = set()
    hyperParams = set()
    dimensions = set()
    try:
        search_path = os.path.join('..', 'cluster', 'graphics', '*', type_query, '*', '*', '*', '*', '*')
        print(search_path)
        items = glob.glob(search_path) # os.listdir(f'../cluster/graphics/*/{type_query}')
        for path in items:
            temp = path.split('\\')
            dataCategories.add(temp[3])
            pitchTypes.add(temp[-5])
            standardizes.add(temp[-4])
            groupings.add(temp[-3])
            dimensions.add(temp[-2])
            hyperParams.add(temp[-1].removesuffix('.html'))

        # print(items)
    except Exception as e:
        print(e)
        print('shit')
    print(dataCategories)
    print(pitchTypes)
    print(standardizes)
    print(groupings)
    print(dimensions)
    print(hyperParams)

    return jsonify({
            'dataCategories':list(dataCategories),
            'pitchTypes':list(pitchTypes),
            'standardizes':list(standardizes),
            'groupings':list(groupings),
            'dimensions':list(dimensions),
            'hyperParams':list(hyperParams)

        })

if __name__ == '__main__':
    app.run(debug=True)
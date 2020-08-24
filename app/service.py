import argparse, logging, os
from flask import Flask, jsonify, request, render_template
from datetime import datetime
from search import SearchIndex, SearchComponent
import multiprocessing as mp

app = Flask(__name__)
PORT = 13000

DATAFILE = 'items.snappy.avro'
DATADIR = 'data'
N_RECORDS = 3000000

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def get_similar_items():
    global search_component
    body = request.form
    itemId, category, numResults, imageUrl = int(body["itemId"]), body.get("category"), int(body["numOfResults"]), None
    if category is not None:
        category = int(category)
    results = []
    start = datetime.now()
    for itId, similarity, url in search_component.query(itemId, category=category, numResults=numResults):
        if itId == itemId:
            imageUrl = url
        else:
            results.append({"itemId": itId, "imageUrl": url, "score": similarity})
    requestTime = int((datetime.now() - start).total_seconds() * 1000)
    
    response = {
        "item": {
            "itemId": itemId,
            "category": category,
            "imageUrl": imageUrl
        },
        "results": results[:numResults],
        "time": requestTime
    }
    return jsonify(response)

@app.route('/getRandom', methods=['GET'])
def get_random_items():
    global search_component
    n = int(request.args['n'])
    try:
        category = int(request.args.get('category'))
    except:
        category = None
    items = search_component.getNRandom(n, category=category)
    return jsonify({"results": items})

if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--njobsindex', metavar='n_jobs_index', type=int, default=0,\
                        help='number of processes to use for building search index (0 to use all CPU cores)')
    parser.add_argument('--njobscomp', metavar='n_jobs_component', type=int, default=0,\
                        help='number of processes to use for building search component (0 to use all CPU cores)')
    parser.add_argument('--indexloadpath', '-iload', metavar='index_load_path', type=str, default=None,\
                        help='path to load a saved search index from')
    parser.add_argument('--comploadpath', '-cload', metavar='comp_load_path', type=str, default=None,\
                        help='path to load a saved search component from')
    parser.add_argument('--indexsavepath', '-isave', metavar='index_save_path', type=str, default=None,\
                        help='path to save search index to be loaded in future runs')
    parser.add_argument('--compsavepath', '-csave', metavar='comp_save_path', type=str, default=None,\
                        help='path to save search component to be loaded in future runs')
    parser.add_argument('--datadir', metavar='datadir', type=str, default=DATADIR,\
                        help='directory to store data for search index and component in')
    parser.add_argument('--datafile', metavar='datafile', type=str, default=DATAFILE,\
                        help='path to avro file containing item data for searching')
    parser.add_argument('--verbose', '-v', action='count', default=4,\
                        help='Verbosity level for logging')
    args = parser.parse_args()

    levels = {1: logging.CRITICAL, 2: logging.ERROR, 3: logging.WARNING, 4: logging.INFO,\
              5: logging.DEBUG, 0: logging.NOTSET}
    logger = logging.getLogger('similar_item_service')
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=levels.get(args.verbose, logging.NOTSET), datefmt="%H:%M:%S")

    with (SearchIndex(args.datafile, datadir=args.datadir, n_records=N_RECORDS, n_jobs=args.njobsindex) if args.indexloadpath is None\
          else SearchIndex.load(args.indexloadpath)) as search_index:
        if args.indexsavepath is not None:
            os.makedirs(os.path.basename(args.indexsavepath), exist_ok=True)
            search_index.save(args.indexsavepath)
        with (SearchComponent(search_index, datadir=args.datadir, n_jobs=args.njobscomp) if args.comploadpath is None\
            else SearchComponent.load(args.comploadpath, search_index)) as search_component:
            if args.compsavepath is not None:
                os.makedirs(os.path.basename(args.compsavepath), exist_ok=True)
                search_component.save(args.compsavepath)
            logger.info("Starting service")
            app.run(host='0.0.0.0', debug=False, port=PORT)
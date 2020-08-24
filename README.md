# Similar Item Search Service
To use the service, clone this repo and then place the avro items data file in the *'app/data'* directory.

Build the docker image by running in a terminal:

	docker build --tag similar-item-search .

The more resources available in the docker environment, the better the service will perform. 4+ GB of RAM is recommended.

Start the app by running:

	docker run --name [container-name] -p 13000:13000 similar-item-search

along with any of the following optional arguments:

		--njobsindex: number of processes to use for building search index (default=0, use all CPU cores)
		--njobscomp: number of processes to use for building search component (default=0, use all CPU cores)
		--indexloadpath: path to load a saved search index from
		--comploadpath: path to load a saved search component from
		--indexsavepath: path to save search index after building, to be loaded in future runs
		--compsavepath: path to save search component to be loaded in future runs
		--datadir: directory to store data for search index and component in (avro database file must be present in this directory)
		--datafile: path to avro database file within datadir
		--verbose, -v (count): Verbosity level for logging


It will take up to a few hours to build the search index and component. Once this is done, the server will start and HTTP POST requests for similar item queries can be made in the format:

		{
			"itemId": 1234567 , // Query item id
			"category": 2 , // Query item category
			"numOfResults": 100 // Number of similar items expected in the response
		}


The user interface can be accessed from a browser at <localhost:13000>. The page will display random items on start. Clicking on an item will populate the query input box with the item's id. A query can be launched by switching the query mode selector to *'similar items'* and inputting an item id and a number of similar items to display, and clicking on the *'Go'* button.

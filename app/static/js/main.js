var categoryMap = {
  15687: "Men's T-Shirts",
  15709: "Men's Athletic Shoes",
  63861: "Women's Dresses",
  169291: "Women's Bags & Handbags"
};

$(document).on({
    ajaxStart: function(){
        $("body").addClass("loading"); 
    },
    ajaxStop: function(){ 
        $("body").removeClass("loading"); 
    }    
});

$(document).ready(function() {
	fillCategories();
	$("#modes").change(function() {
		var mode = $("#modes").children("option:selected").val();
		switch(mode) {
			case "random":
				$("#itemId").prop('disabled', true);
				break;
			case "similar":
				$("#itemId").prop('disabled', false);
				break;
		}
	});
	$("#goButton").click(updateItems);
	updateItems();
});

function fillCategories() {
	for(var catId in categoryMap) {
		var catText = categoryMap[catId];
		$('#categories').append(new Option(catText, catId));
	}
}

function updateItems() {
	var mode = $("#modes").children("option:selected").val();
	var itemId = $("#itemId").val();
	var n = $("#nItems").val();
	var category = $("#categories").children("option:selected").val();
	var request = null;
	var params = {};
	switch(mode) {
	  case "random":
	    request = 'http://localhost:13000/getRandom';
	    params["n"] = n;
	    if (category in categoryMap) {
	    	params["category"] = category;
	    }
	    $.get(request, params, fillItems);
	    break;
	  case "similar":
	    request = 'http://localhost:13000/query';
	    params["itemId"] = itemId;
	    if (category in categoryMap) {
	    	params["category"] = category;
	    }
	    params["numOfResults"] = n;
	    $.post(request, params, fillItems);
	    break;
	}
}

function fillItems(response) {
	$('#items').empty();
	response["results"].forEach(function(item) {
		var itemDiv = $("<div class='item'></div>");
		itemDiv.attr("id", "item-" + item["itemId"]);
		var imgHtml = "<div class='square itemLink'><img class='itemImage' src='" + item["imageUrl"] + "'></div>";
		itemDiv.append(imgHtml);
		var imgDiv = itemDiv.children().last();
		imgDiv.click(function(e) {
			var itemId = $(this).parent().attr('id').split('-');
			itemId = parseInt(itemId[itemId.length-1]);
			$("#itemId").val(itemId);
		});
		var capHtml = "<p>" + ("category" in item ? categoryMap[item["category"]] : Math.round(item["score"]*100) + "% similar") + "</p>";
		itemDiv.append(capHtml);
		$('#items').append(itemDiv);
	});
}
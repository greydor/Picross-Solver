document.addEventListener("DOMContentLoaded", function() {
    
    let id_button = document.querySelector("#id_button");
    id_button.addEventListener("click", function() {
        let size = document.querySelector("#puzzle_id").value;
    })

    let random_button = document.querySelector("#random_button");
    random_button.addEventListener("click", function() {
        let id = document.querySelector("#puzzle_size").value;
        debugger;
    })    
})
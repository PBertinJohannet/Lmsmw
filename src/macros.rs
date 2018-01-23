//! Module for the layer macro
//! the layers macro creates layer config with sigmoid activations.
//! The activation functions can be changed later.
#[macro_export]
macro_rules! layers {
    ( $( $x:expr),* ) => {
        {
            use $crate::network::LayerConfig;
            let mut temp_vec = Vec::new();
            $( temp_vec.push(LayerConfig::new($x));)*
            temp_vec
        }
    }
}

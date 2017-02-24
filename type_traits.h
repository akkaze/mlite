#ifndef MLITE_TYPE_TRAITS_H_
#define MLITE_TYPE_TRAITS_H_

namespace mlite {
enum ReturnType {
	kSimple = 0,
	kInplace = 1,
};
}
namespace mlite {
//@brief macro to quickly declare traits information */
#define MLITE_DECLARE_TRAITS(Trait, Type, Value)      \
  template<>                                          \
  struct Trait<Type> {                                \
    static const bool value = Value;                  \
  }													  
template<ReturnType return_type>
struct is_not_inplace {
	static const bool value = true;
};
MLITE_DECLARE_TRAITS(is_not_inplace, ReturnType::kInplace, false);

//@brief simple enable_if implementation,here we don't trigger SFINAE but typedef void
template<bool Cond,typename T>
struct enable_if { typedef T type; };

template<typename T>
struct enable_if<false, T> { typedef void type; };
}
#endif // !MLITE_TYPE_TRAITS_H_
